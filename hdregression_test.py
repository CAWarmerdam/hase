#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import os
import sys

import h5py
import tables

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from scipy import stats

from hdgwas.hdregression import A_covariates, A_tests, B_covariates, C_matrix, B4, A_inverse, HASE, \
    calculate_variant_dependent_A, A_inverse_2, HASE2


class EncoderCopy:
    def __init__(self, number_of_individuals):
        self.F = np.random.randint(
            1, 10, number_of_individuals * number_of_individuals).reshape(
            number_of_individuals, number_of_individuals)
        self.F_inv = np.linalg.inv(self.F)

    def encode_genotype_matrix(self, genotype_matrix):
        return np.dot(genotype_matrix, self.F)

    def encode_with_inverse(self, phenotype_matrix):
        return np.dot(self.F_inv, phenotype_matrix)


def get_genotype_matrix(number_of_individuals, number_of_variants):
    return np.random.randint(
        0, 3, number_of_variants * number_of_individuals)\
        .reshape(number_of_variants, number_of_individuals)


def get_covariates_matrix(number_of_individuals):
    return np.random.randint(0, 4, number_of_individuals).reshape(number_of_individuals, 1)


def get_phenotype_matrix_with_interaction(X):
    reshape = np.random.normal(0, 2, X.shape[0] * (X.shape[2])).reshape(X.shape[0], X.shape[2])
    return X[:,3] * 4 + X[:,2] * 0.5 + X[:,1] * 3 + 10 + reshape


def get_phenotype_matrix_no_interaction(X):
    reshape = np.random.normal(0, 2, X.shape[0] * (X.shape[2])).reshape(X.shape[0], X.shape[2])
    return X[:,2] * 4 + X[:,1] * 3 + 10 + reshape


def fit_model(X2, y):
    X2 = X2[:,1:]
    lm = linear_model.LinearRegression()
    lm.fit(X2, y)

    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X2)

    newX = pd.DataFrame({"Constant": np.ones(len(X2))}).join(pd.DataFrame(X2))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    # Note if you don't want to use a DataFrame replace the two lines above with
    # newX = np.append(np.ones((len(X),1)), X, axis=1)
    # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b,
                                                                                                 p_values]
    return myDF3


def calculate_a(covariates_matrix, genotype_matrix):
    a_covariates = A_covariates(covariates_matrix)
    a_tests = A_tests(covariates_matrix, genotype_matrix)
    return A_inverse(a_covariates, a_tests)


def test_hase2_no_interaction(number_of_individuals,
                              number_of_variants):
    genotype_matrix = get_genotype_matrix(number_of_individuals,
                                          number_of_variants)
    covariates_matrix = get_covariates_matrix(number_of_individuals)
    covariates_matrix_extended = covariates_matrix.repeat(number_of_variants, axis=1)
    base_x = np.stack((covariates_matrix_extended, genotype_matrix.T), axis=1)
    matrix_with_determinants_x = get_x_no_interaction(base_x)

    phenotype_matrix = get_phenotype_matrix_no_interaction(matrix_with_determinants_x)[:, 0:2]

    # Encode the data
    encoder = EncoderCopy(number_of_individuals)
    genotype_matrix_encoded = encoder.encode_genotype_matrix(genotype_matrix)
    phenotype_matrix_encoded = encoder.encode_with_inverse(phenotype_matrix)

    # Calculate the A inverse the new way
    a_inverse_alternative = calculate_a_alternative(covariates_matrix, genotype_matrix)

    # Calculate the A inverse the old way
    a_inverse = calculate_a(covariates_matrix, genotype_matrix)

    b_cov = B_covariates(covariates_matrix, phenotype_matrix)

    C = C_matrix(phenotype_matrix)

    b4 = B4(phenotype_matrix_encoded, genotype_matrix_encoded)

    b_variable = b4[np.newaxis, ...]

    number_of_variable_terms = b_variable.shape[0]

    N_con = a_inverse_alternative.shape[1] - number_of_variable_terms

    DF = (number_of_individuals - a_inverse_alternative.shape[1])

    t_stat, SE = HASE(b4, a_inverse, b_cov, C, N_con, DF)

    t_stat2, SE2 = HASE2(b_variable, a_inverse_alternative, b_cov, C, N_con, DF)

    assert(np.allclose(SE, SE2))
    assert(np.allclose(t_stat, t_stat2))
    print("Standard error and t-statistics equal between old hase and new hase")

    model_table = fit_model(matrix_with_determinants_x[..., 0], phenotype_matrix[..., 0])

    assert(np.isclose(model_table.loc[a_inverse_alternative.shape[1] - 1, u"Standard Errors"], SE[0, 0, 0]))
    assert(np.isclose(model_table.loc[a_inverse_alternative.shape[1] - 1, u"t values"], t_stat[0, 0, 0]))
    print("Standard error and t-statistics equal between hase and regular regression analysis")


def calculate_a_alternative(covariates_matrix, genotype_matrix):
    # Get empty matrix of covariate values
    factor_matrix = np.empty((0, 0))
    # Get the constant part of A
    a_cov = A_covariates(covariates_matrix)
    variant_dependent_A = calculate_variant_dependent_A(genotype_matrix,
                                                        factor_matrix,
                                                        covariates_matrix)
    a_inverse_alternative = A_inverse_2(a_cov, variant_dependent_A)
    return a_inverse_alternative


def get_x_no_interaction(base_x):
    x = np.stack([preprocessing.add_dummy_feature(base_x[..., i]) for i in range(base_x.shape[-1])],
                 axis=2)
    return x


def get_x_with_interaction(base_X):
    poly = PolynomialFeatures(interaction_only=True)
    X = np.stack([poly.fit_transform(base_X[..., i])[:, [0, 1, 3, 2]] for i in range(base_X.shape[-1])],
                             axis=2)
    return X


def test_hase2_with_interaction(number_of_individuals, number_of_variants):
    # Get a genotype matrix
    genotype_matrix = get_genotype_matrix(number_of_individuals,
                                          number_of_variants)
    # Get the matrix of covariates
    covariates_matrix = get_covariates_matrix(number_of_individuals)
    # Get a matrix of covariates that is extended to be the same shape as the
    # genotype matrix
    covariates_matrix_extended = covariates_matrix.repeat(number_of_variants, axis=1)
    # Create the X matrix with base determinants genotype and the covariate
    base_X = np.stack((covariates_matrix_extended, genotype_matrix.T), axis=1)
    # Get the X matrix with the
    X = get_x_with_interaction(base_X)
    # Get the phenotype matrix
    phenotype_matrix = get_phenotype_matrix_with_interaction(X)[:, 0:2]

    # Calculate the phenotypes * interaction covariate
    interaction_phenotype_matrix = np.multiply(phenotype_matrix, covariates_matrix)

    # Encode the stuff
    encoder = EncoderCopy(number_of_individuals)
    genotype_matrix_encoded = encoder.encode_genotype_matrix(genotype_matrix)
    phenotype_matrix_encoded = encoder.encode_with_inverse(phenotype_matrix)
    interaction_phenotype_matrix_encoded = encoder.encode_with_inverse(
        interaction_phenotype_matrix)

    # Get the constant part of A
    a_cov = A_covariates(covariates_matrix)
    variant_dependent_A = calculate_variant_dependent_A(genotype_matrix,
                                                        covariates_matrix,
                                                        covariates_matrix)

    # Get the b part that corresponds to the interaction values
    b_interaction = np.dot(genotype_matrix_encoded, interaction_phenotype_matrix_encoded)

    b_cov = B_covariates(covariates_matrix, phenotype_matrix)

    C = C_matrix(phenotype_matrix)

    b4 = B4(phenotype_matrix_encoded, genotype_matrix_encoded)

    a_inv = A_inverse_2(a_cov, variant_dependent_A)

    b_variable = np.stack((b_interaction, b4))

    number_of_variable_terms = b_variable.shape[0]

    N_con = a_inv.shape[1] - number_of_variable_terms

    DF = (number_of_individuals - a_inv.shape[1])

    t_stat, SE = HASE2(b_variable, a_inv, b_cov, C, N_con, DF)

    model_table = fit_model(X[..., 0], phenotype_matrix[..., 0])

    # Assert whether the HASE2 method returns equal results compared to the regular
    # regression analyses.
    assert(np.isclose(model_table.loc[a_inv.shape[1] - 1, u"Standard Errors"], SE[0, 0, 0]))
    assert(np.isclose(model_table.loc[a_inv.shape[1] - 1, u"t values"], t_stat[0, 0, 0]))
    print("Standard error and t-statistics equal between hase and regular regression analysis")


def main(argv=None):
    if argv is None:
        argv = sys.argv

    number_of_individuals = 8
    number_of_variants = 3

    test_hase2_no_interaction(number_of_individuals, number_of_variants)

    test_hase2_with_interaction(number_of_individuals, number_of_variants)
    return 0


if __name__ == "__main__":
    sys.exit(main())