from __future__ import print_function
import numpy as np
import os
import sys
from tools import Timer, timer, timing, save_parameters
import scipy.linalg.blas as FB
import h5py
import gc
import tables


# @timing
def A_covariates(covariates, intercept=True):
    '''
    :param covariates: (n_subjects, n_covariates) - only constant covariates should be included (age, sex, ICV etc)
    :param intercept: default True, add intercept to model
    :return: matrix (n_cavariates, n_covariates), constant part for the rest of the study
    '''

    S, N = covariates.shape
    if intercept:
        I = np.ones(S).reshape(S, 1)
        covariates = np.hstack((I, covariates))
    a_cov = np.dot(covariates.T, covariates)
    return a_cov


# @timing
def B4(phenotype, genotype):
    b4 = np.tensordot(genotype, phenotype, axes=([1], [0]))
    return b4


def interaction(genotype, factor):
    g = genotype * factor.T
    return g


def calculate_interaction_b(genotype, interaction_values):
    """
    Decodes genotype and interaction values into the sum of Y[i] * COV[i] * SNP[i]
    :param genotype: The encoded genotype matrix
    :param interaction_values: The encoded
    :return:
    """
    b_interactions = np.tensordot(genotype, interaction_values, axes=([1], [0]))
    return b_interactions


# @timing
def A_tests(covariates, genotype, intercept=True):  # TODO (low) extend for any number of tests in model
    '''
    :param covariates: (n_subjects, n_covariates) - only constant covariates should be included (age, sex, ICV etc)
    :param genotype: (n_tests, n_subjects) - test could be any kind of quantitative covariance
    :return: (1,n_covariates + intercept)
    '''

    if intercept:
        fst = np.sum(genotype, axis=1).reshape(-1, 1)
        sec = np.dot(genotype, covariates)
        tr = np.sum(np.power(genotype, 2), axis=1).reshape(-1, 1)
        return np.hstack((fst, sec, tr))

    else:
        sec = np.dot(genotype, covariates)
        tr = np.sum(np.power(genotype, 2), axis=1).reshape(-1, 1)
        return np.hstack((sec, tr))


def calculate_variant_dependent_A(genotype, factor_matrix,
                                  covariates, intercept=True):
    number_of_variable_terms = (factor_matrix.shape[1] + 1)
    number_of_total_terms = (covariates.shape[1] + number_of_variable_terms)
    if intercept:
        number_of_total_terms += 1

    variant_dependent_A = np.zeros(
        number_of_variable_terms * number_of_total_terms * genotype.shape[0]).reshape((
        number_of_variable_terms, number_of_total_terms, genotype.shape[0]))
    variable_term_index = 0
    # Have to add extra columns to the covariates

    covariates = np.tile(covariates, (genotype.shape[0],1,1))

    # Loop through the columns of the factor matrix
    for factor_column in factor_matrix.T:
        # Multiply the genotypes with the factor column
        interaction_values = genotype * factor_column
        # This creates a 2d array with columns representing the
        # variants and values representing the covariates from individuals

        # Calculate the A values for interaction values and other independent
        # determinants already in covariates matrix
        sec = calculate_dot_product_for_variants(covariates, interaction_values)
        tr = np.sum(np.power(interaction_values, 2), axis=1).reshape(-1, 1)

        if intercept:
            fst = np.sum(interaction_values, axis=1).reshape(-1, 1)
            variable_term_values = np.hstack((fst, sec, tr)).T
            variant_dependent_A[variable_term_index, 0:variable_term_values.shape[0]] = variable_term_values
        else:
            variable_term_values = np.hstack((sec, tr)).T
            variant_dependent_A[variable_term_index, 0:variable_term_values.shape[0]] = variable_term_values

        variable_term_index += 1
        covariates = np.dstack((covariates, interaction_values))

    # Calculate the A values for genotypes with the other independent determinants
    sec = calculate_dot_product_for_variants(covariates, genotype)
    tr = np.sum(np.power(genotype, 2), axis=1).reshape(-1, 1)

    if intercept:
        fst = np.sum(genotype, axis=1).reshape(-1, 1)
        variable_term_values = np.hstack((fst, sec, tr)).T
        variant_dependent_A[variable_term_index, 0:variable_term_values.shape[0]] = variable_term_values
    else:
        variable_term_values = np.hstack((sec, tr)).T
        variant_dependent_A[variable_term_index, 0:variable_term_values.shape[0]] = variable_term_values
    return variant_dependent_A.T


def calculate_dot_product_for_variants(covariates, other_independent_determinant):
    # In the dot einsum notation, labels represent the following:
    # i: variants
    # j: individuals (dot product of genotypes, covariates)
    # k: different covariates
    sec = np.einsum('ij,ijk->ik', other_independent_determinant, covariates)
    # (Values get summed along individuals)
    return sec


# @timing
def B_covariates(covariates, phenotype, intercept=True):
    S, N = covariates.shape

    b_cov = np.dot(covariates.T, phenotype)
    if intercept:
        b1 = np.sum(phenotype, axis=0).reshape(1, phenotype.shape[1])
        B13 = np.append(b1, b_cov, axis=0)
        return B13
    else:
        return b_cov


def A_inverse_2(a_covariates, variant_dependent_a):
    A_inv = []
    n,m = a_covariates.shape
    k = n + variant_dependent_a.shape[2]

    for i in range(variant_dependent_a.shape[0]):
        inv = np.zeros(k*k).reshape(k,k)
        inv[0:n, 0:n] = a_covariates
        inv[:, n:k] = variant_dependent_a[i,:]
        inv[n:k, :k-1] = np.maximum(inv[n:k, :k-1],
                                    variant_dependent_a[i,:k-1].T)
        try:
            A_inv.append(np.linalg.inv(inv))
        except:
            A_inv.append(np.zeros(k * k).reshape(k, k))  # TODO (high) test; check influence on results; warning;
    return np.array(A_inv)


# @timing
def A_inverse(a_covariates, a_test):  # TODO (low) extend for any number of tests in model

    A_inv = []
    n, m = a_covariates.shape
    k = n + 1
    for i in xrange(a_test.shape[0]):  # TODO (low) not in for loop
        inv = np.zeros(k * k).reshape(k, k)
        inv[0:k - 1, 0:k - 1] = a_covariates
        inv[k - 1, :] = a_test[i, :]
        inv[0:k, k - 1] = a_test[i, 0:k]
        try:
            A_inv.append(np.linalg.inv(inv))
        except:
            A_inv.append(np.zeros(k * k).reshape(k, k))  # TODO (high) test; check influence on results; warning;

    return np.array(A_inv)


# @timing
def C_matrix(phenotype):
    C = np.einsum('ij,ji->i', phenotype.T, phenotype)
    return C


# @timing
# @save_parameters
def HASE(b4, A_inverse, b_cov, C, N_con, DF):
    with Timer() as t:
        # These together form the X matrix
        B13 = b_cov
        B4 = b4

        A1_B_constant = np.tensordot(A_inverse[:, :, 0:(N_con)], B13, axes=([2], [0]))

        A1_B_nonconstant = np.einsum('ijk,il->ijl', A_inverse[:, :, N_con:N_con + 1], B4)

        # Combine the inverse of A multiplied with
        # constant part of B and non-constant part of B
        A1_B_full = A1_B_constant + A1_B_nonconstant

        BT_A1B_const = np.einsum('ij,lji->li', B13.T, A1_B_full[:, 0:(N_con), :])

        BT_A1B_nonconst = np.einsum('ijk,ijk->ijk', B4[:, None, :], A1_B_full[:, (N_con):N_con + 1, :])

        BT_A1B_full = BT_A1B_const[:, None, :] + BT_A1B_nonconst

        C_BTA1B = BT_A1B_full - C.reshape(1, -1)

        C_BTA1B = np.abs(C_BTA1B)

        a44_C_BTA1B = C_BTA1B * A_inverse[:, (N_con):N_con + 1, (N_con):N_con + 1]

        a44_C_BTA1B = np.sqrt((a44_C_BTA1B))

        t_stat = np.sqrt(DF) * np.divide(A1_B_full[:, (N_con):N_con + 1, :], a44_C_BTA1B)

        SE = a44_C_BTA1B / np.sqrt(DF)

    print("time to compute GWAS for {} phenotypes and {} SNPs .... {} sec".format(b4.shape[1],
                                                                                  A_inverse.shape[0],
                                                                                  t.secs))
    return t_stat, SE


# @timing
# @save_parameters
def HASE2(b_variable, a_inverse, b_cov, C, number_of_constant_terms, DF):
    with Timer() as t:
        # These together form the X matrix
        B13 = b_cov
        number_of_variable_terms = b_variable.shape[0]
        variant_effect_index = a_inverse.shape[1] - 1

        A1_B_constant = np.tensordot(a_inverse[:, :, 0:(number_of_constant_terms)], B13, axes=([2], [0]))

        # In the einsum notation, the labels represent the following:
        # i: The variant axis
        # j: ...
        # k: The axis with regression terms (interaction, genotype)
        # l: The phenotype axis
        A1_B_nonconstant = np.einsum('ijk,kil->ijl',
                                     a_inverse[:, :, number_of_constant_terms:number_of_constant_terms + number_of_variable_terms],
                                     b_variable)

        # Combine the inverse of A multiplied with
        # constant part of B and non-constant part of B
        A1_B_full = A1_B_constant + A1_B_nonconstant

        # In the einstein summation notation, the labels represent the following:
        # l: The variant axis
        # j: The the axis with the regression terms (interaction, genotype)
        # i: The phenotype axis
        BT_A1B_const = np.einsum('ij,lji->li', B13.T, A1_B_full[:, :(number_of_constant_terms), :])

        # In the einstein summation notation, the labels represent the following
        # i: The variant axis
        # j: Terms
        # k: Phenotype
        BT_A1B_nonconst = np.einsum(
            'ijk,ijk->ik', b_variable.transpose((1,0,2)),
            A1_B_full[:, (number_of_constant_terms):number_of_constant_terms + number_of_variable_terms, :])

        # Combine the constant and nonconstant parts of the BT, beta matrix
        BT_A1B_full = BT_A1B_const + BT_A1B_nonconst

        # Get the difference between C (dot product of phenotypes) and the matrix of estimates
        C_BTA1B = BT_A1B_full - C.reshape(1, -1)
        C_BTA1B = np.abs(C_BTA1B)

        # Multiply the far right / lower part of every A_inv matrix
        # (this is the part with the sum of the squares of genotype dosages),
        # with the differences between C and the BT_A1B_full matrix
        a44_C_BTA1B = np.einsum(
            'il,i...->i...l', C_BTA1B,
            a_inverse[:, variant_effect_index, (variant_effect_index):variant_effect_index+1])

        # Square this result
        a44_C_BTA1B = np.sqrt((a44_C_BTA1B))

        # Get the t-statistics
        t_stat = np.sqrt(DF) * np.divide(
            A1_B_full[:, (variant_effect_index):variant_effect_index+1, :], a44_C_BTA1B)
        # The t-statistics are stored in a 3d array with the first dimension
        # representing the variants (main determinants to test), the second
        # dimension being synonymous (or at least as it appears to me) to the
        # third dimension representing phenotypes.
        # Second dimension thus only contains one element (phenotype array)

        # Get the standard error.
        SE = a44_C_BTA1B / np.sqrt(DF)
        # Standard errors are stored in the same ways as the t-statistics:
        # SE[variant][0][phenotype]

    print("time to compute GWAS for {} phenotypes and {} SNPs .... {} sec".format(b_variable.shape[1],
                                                                                  a_inverse.shape[0],
                                                                                  t.secs))
    return t_stat, SE
