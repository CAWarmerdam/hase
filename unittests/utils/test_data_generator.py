#!/usr/bin/env python3

"""
Test script for the classic meta analysis
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

# constants
from sklearn import linear_model

from unittests.utils.hase_h5_writer import GenotypeData, HaseHDF5Writer
from unittests.test_hdregression import fit_model

DEFAULT_NUMBER_OF_VARIANTS = 128
DEFAULT_NUMBER_OF_PHENOTYPES = 48


def sample_variants_from_ref():
    ref_data_set = pd.read_csv(
        "~/Documents/hdaa/hase_repository/data/1000Gp1v3.ref.gz", sep=" ",
        compression="gzip")
    variants_data_frame = ref_data_set.sample(n=DEFAULT_NUMBER_OF_VARIANTS)
    return variants_data_frame


def fit_model(X2, y):
    lm = linear_model.LinearRegression()
    lm.fit(X2, y)

    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X2)

    newX = pd.DataFrame({"Constant": np.ones(len(X2))}).join(pd.DataFrame(X2))
    MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX.columns))

    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"] = [params, sd_b]
    return myDF3


class TestDataException(Exception):
    pass


class TestDataGenerator:

    def __init__(self,
                 variant_data_frame, number_of_phenotypes,
                 base_output_path, cohort_prefix,
                 number_of_individuals_per_cohort=(200, 300, 400)):
        self.cohort_names = [cohort_prefix + "_" + str(i) for i in range(len(number_of_individuals_per_cohort))]
        self.base_output_path = base_output_path
        self.genotype_matrices = None
        self.phenotype_matrix = None
        self.independent_variable_matrix = None
        self.number_of_individuals_per_cohort = {self.cohort_names[i]: n for i, n in enumerate(number_of_individuals_per_cohort)}
        self.number_of_individuals = sum(number_of_individuals_per_cohort)
        self.number_of_variants = variant_data_frame.shape[0]
        self.number_of_phenotypes = number_of_phenotypes
        self.number_of_covariates = 2
        variant_data_frame.reset_index(drop=True, inplace=True)
        self.variant_data_frame = variant_data_frame
        self.phenotype_names = np.array(["pheno_{}".format(i) for i in range(self.number_of_phenotypes)])

    def get_individual_ids(self, lower_bound, upper_bound):
        return np.array(["S{}".format(i) for i in range(lower_bound, upper_bound)])

    def generate_test_data(self):
        genotype_matrix = self.generate_genotype_matrix()
        covariates_matrix = self.generate_covariate_matrix()  # One covariate
        covariates_matrix_extended = np.repeat(
            covariates_matrix[:, :, np.newaxis],
            self.number_of_variants,
            axis=2).transpose((1, 0, 2))

        independent_variable_matrix = np.concatenate((covariates_matrix_extended, genotype_matrix.T[np.newaxis, :, :]),
                                                     axis=0)
        phenotype_matrix = self.get_phenotype_matrix_no_interaction(independent_variable_matrix)

        self.independent_variable_matrix = independent_variable_matrix
        self.phenotype_matrix = phenotype_matrix

    def generate_genotype_matrix(self):
        return np.random.randint(
            0, 3, self.number_of_variants * self.number_of_individuals) \
            .reshape(self.number_of_variants, self.number_of_individuals) \
            .astype(np.float)

    def generate_covariate_matrix(self):
        return np.random.randint(0, 4, self.number_of_individuals * self.number_of_covariates) \
            .reshape(self.number_of_individuals, self.number_of_covariates)

    def get_phenotype_matrix_no_interaction(self, independent_variable_matrix):

        # Introduce an effect of covariates
        covariate_effects = np.random.normal(0, 0.5, self.number_of_phenotypes * self.number_of_covariates).reshape(
            self.number_of_covariates, self.number_of_phenotypes)
        covariate_term = np.einsum('cs,cp -> sp', independent_variable_matrix[0:self.number_of_covariates, :, 0],
                                   covariate_effects)

        # Introduce an effect of genetics
        genotype_effects = np.random.normal(0, 0.5, self.number_of_phenotypes * self.number_of_variants).reshape(
            self.number_of_variants, self.number_of_phenotypes)
        genotype_term = np.einsum(
            'sg,gp -> sp',
            independent_variable_matrix[self.number_of_covariates, :, :],
            genotype_effects)

        # Introduce a baseline effect (intercept)
        constant_term = np.random.normal(0, 2, self.number_of_individuals * self.number_of_phenotypes) \
            .reshape(self.number_of_individuals, self.number_of_phenotypes)

        # We want to create an error term so that the stuff does not fit perfectly
        error_term = np.random.normal(0, 2, self.number_of_individuals * self.number_of_phenotypes) \
            .reshape(self.number_of_individuals, self.number_of_phenotypes)

        return np.sum(np.stack((constant_term, genotype_term, covariate_term, error_term), axis=0), axis=0)

    def get_genotype_data(self, variant_indices):
        matrix = self.independent_variable_matrix[-1, ...].T
        genotype_data_list = dict()

        individuals_cumulative = 0

        for cohort_name, number_of_individuals in self.number_of_individuals_per_cohort.items():
            cohort_indices = variant_indices.loc[cohort_name].values
            sliced_indices = cohort_indices[cohort_indices != -1]

            # We need to know now where in the cohort space,
            # -1 are located. These columns should not get assigned a variant.
            # We want to remove these columns.
            # We can easily do this by checking which columns have not been assigned.
            # The theoretical indices that are not in the cohort indices are the
            # column indices that will not get assigned. These we should remove.
            non_missing_variants = np.intersect1d(np.arange(
                cohort_indices.shape[0]),
                sliced_indices)

            # We need to make a new matrix of the dimensions that is correct for
            # the cohort in question. Now this also includes the missing variants,
            # later, we remove these.
            output_matrix = np.empty((cohort_indices.shape[0], number_of_individuals))

            individuals_upper_bound = individuals_cumulative + number_of_individuals
            output_matrix[sliced_indices, :] = matrix[
                                               cohort_indices != -1,
                                               individuals_cumulative:individuals_upper_bound]

            if not np.all(np.equal(
                output_matrix[sliced_indices, :],
                    matrix[cohort_indices != -1, individuals_cumulative:individuals_upper_bound])):
                raise TestDataException

            output_variant_data_frame = self.variant_data_frame.copy()
            output_variant_data_frame.iloc[sliced_indices, :] = self.variant_data_frame.loc[cohort_indices != -1].values

            if not np.all(np.equal(
                output_variant_data_frame["ID"].iloc[sliced_indices].to_numpy(),
                    self.variant_data_frame.loc[cohort_indices != -1, "ID"].to_numpy())):
                raise TestDataException

            genotype_data_list[cohort_name] = GenotypeData(
                genotype_matrix=output_matrix[non_missing_variants,:],
                variant_data_frame=output_variant_data_frame.iloc[non_missing_variants, :],
                individual_ids=self.get_individual_ids(individuals_cumulative, individuals_upper_bound))

            individuals_cumulative += number_of_individuals

        return genotype_data_list

    def _generate_missingness_indices_for_cohort(self, size, missingness_rate=0.05, keep_order=True):

        if keep_order:
            indices = np.arange(size)
        else:
            indices = np.random.choice(
                np.arange(size),
                size=size, replace=False)

        if missingness_rate != 0:
            missingness = np.random.choice(
                (True, False),
                size=size,
                p=(missingness_rate, 1 - missingness_rate))

            indices[missingness] = -1

        return indices

    def generate_variant_indices(self, missingness_rate=None):
        names = self.variant_data_frame['ID'].values
        array = self.generate_missingness_indices(names, missingness_rate, keep_order=False)
        return array

    def generate_phenotype_indices(self, missingness_rate=None):
        array = self.generate_missingness_indices(
            self.phenotype_names, missingness_rate, keep_order=False)
        return array

    def generate_missingness_indices(self, names, missingness_rate=None, keep_order=True):
        if missingness_rate is None:
            missingness_rate = [0.05] * len(self.number_of_individuals_per_cohort)
        if len(missingness_rate) != len(self.number_of_individuals_per_cohort):
            raise Exception("length of missingness rate, not matching number of cohorts")
        indices = list()
        for rate in missingness_rate:
            indices.append(
                self._generate_missingness_indices_for_cohort(
                    len(names), rate, keep_order=keep_order))
        array = pd.DataFrame(np.array(indices), index=self.cohort_names,
                             columns=names)
        return array

    def get_expected_meta_analysis(self):
        pass

    def write_genotype_data(self, variant_indices):
        genotype_data = self.get_genotype_data(variant_indices)

        for cohort_name, data in genotype_data.items():
            HaseHDF5Writer(
                path=os.path.join(self.base_output_path, cohort_name, "genotypes_hdf5"),
                chunk_size=30,
                study_name=cohort_name) \
                .write_genotype_matrix(genotype_data=data)

            data.to_csv(path=os.path.join(self.base_output_path, cohort_name, "genotypes.csv"))

    def write_phenotype_data(self, phenotype_indices):
        phenotype_data = self.get_phenotype_data(phenotype_indices)

        for cohort_name, data in phenotype_data.items():
            phenotypes_folder = os.path.join(self.base_output_path, cohort_name, "phenotypes")
            os.makedirs(phenotypes_folder)
            data.to_csv(os.path.join(phenotypes_folder, cohort_name + ".txt"),
                        sep="\t", index=True, index_label="id")

    def get_phenotype_data(self, phenotype_indices):
        matrix = self.phenotype_matrix
        indices = phenotype_indices

        return self.slice_cohort_data_missingness(indices, matrix, self.phenotype_names)

    def slice_cohort_data_missingness(self, indices, matrix, names):
        # For each column in the matrix, we need to add
        # output matrix
        cohort_data_list = dict()
        individuals_cumulative = 0

        for cohort_name, number_of_individuals in self.number_of_individuals_per_cohort.items():
            cohort_indices = indices.loc[cohort_name].values
            sliced_indices = cohort_indices[cohort_indices != -1]

            non_missing_features = np.intersect1d(np.arange(
                sliced_indices.shape[0]),
                cohort_indices)

            cohort_data = np.empty((number_of_individuals, cohort_indices.shape[0]))
            individuals_upper_bound = individuals_cumulative + number_of_individuals
            cohort_data[:, sliced_indices] = matrix[individuals_cumulative:individuals_upper_bound,
                                             cohort_indices != -1]

            if not np.all(np.equal(
                cohort_data[:, sliced_indices],
                    matrix[individuals_cumulative:individuals_upper_bound, cohort_indices != -1])):
                raise TestDataException

            names_sliced = names.copy()
            names_sliced[sliced_indices] = names[cohort_indices != -1]

            if not np.all(names_sliced[sliced_indices] == names[cohort_indices != -1]):
                raise TestDataException

            cohort_data_list[cohort_name] = pd.DataFrame(
                cohort_data[:, non_missing_features],
                columns=names_sliced[non_missing_features],
                index=self.get_individual_ids(individuals_cumulative, individuals_upper_bound))

            individuals_cumulative += number_of_individuals

        return cohort_data_list

    def write_covariate_data(self):
        matrix = self.independent_variable_matrix[0:self.number_of_covariates,:,0].T
        names = np.array(["cov_{}".format(i) for i in range(matrix.shape[1])])
        indices = self.generate_missingness_indices(
            names,
            missingness_rate=[0]*len(self.number_of_individuals_per_cohort),
            keep_order=True)
        covariate_data = self.slice_cohort_data_missingness(indices, matrix, names)

        for cohort_name, data in covariate_data.items():
            covariates_folder = os.path.join(self.base_output_path, cohort_name, "covariates")
            os.makedirs(covariates_folder)
            data.to_csv(os.path.join(covariates_folder, cohort_name + ".txt"),
                        sep="\t", index=True, index_label="id")

    def write_expected_meta_analysis(self, variant_indices, phenotype_indices):
        """
        Method that writes the expected results for a meta analysis that considers
        missingness in each of the cohorts' variants and phenotypes.
        :param variant_indices:
        :param phenotype_indices:
        :return: None
        """

        # We first have to analyse each of the cohorts separately

        lower_bound_individuals = 0
        upper_bound_individuals = 0

        output_list = list()

        for cohort_name, number_of_individuals in self.number_of_individuals_per_cohort.items():
            upper_bound_individuals += number_of_individuals
            cohort_variant_indices = variant_indices.loc[cohort_name].values
            non_missing_variants = np.where(cohort_variant_indices != -1)[0]
            cohort_phenotype_indices = phenotype_indices.loc[cohort_name].values
            non_missing_phenotypes = np.where(cohort_phenotype_indices != -1)[0]

            cohort_independent_variables = self.independent_variable_matrix[
                                         :, lower_bound_individuals:upper_bound_individuals,...]
            cohort_dependent_variables = self.phenotype_matrix[
                                        lower_bound_individuals:upper_bound_individuals,...]

            output = self.fit_per_cohort_variables(
                cohort_independent_variables, cohort_dependent_variables,
                non_missing_variants, non_missing_phenotypes)

            output.to_csv(
                os.path.join(self.base_output_path, cohort_name, "results.txt"),
                sep="\t", index=True, index_label=["variant", "phenotype"])

            output_list.append(output)

            lower_bound_individuals = upper_bound_individuals

        meta_analysis_results = self.meta_analyse(output_list)
        meta_analysis_results.to_csv(os.path.join(self.base_output_path, "meta.txt"),
                sep="\t", index=True, index_label=["variant", "phenotype"])

        # now meta-analyse
    def fit_per_cohort_variables(self, independent_variables, dependent_variables,
                                 variants_indices, phenotypes_indices):
        out_list = list()
        for variant_index in variants_indices:
            for phenotype_index in phenotypes_indices:
                results = fit_model(
                    independent_variables[..., variant_index].T,
                    dependent_variables[..., phenotype_index]).iloc[[3]]
                results['phenotype'] = self.phenotype_names[phenotype_index]
                results['variant'] = self.variant_data_frame['ID'][variant_index]

                out_list.append(results)

        cohort_results = pd.concat(out_list)
        cohort_results = cohort_results.set_index(keys=["variant", "phenotype"])
        return cohort_results

    def meta_analyse(self, output_list):
        combinations = list(itertools.product(self.variant_data_frame['ID'].values, self.phenotype_names))
        effect_sizes = np.array([output.loc[combinations, "Coefficients"].values for output in output_list])
        standard_error = np.array([output.loc[combinations, "Standard Errors"].values for output in output_list])

        variance = standard_error ** 2

        # Weight the effect sizes by dividing the effect size by
        # the variance of the effect estimate.
        # This is the same thing as multiplying the effect size by
        # (1 / variance).
        effect_size_divided_by_variance = effect_sizes / variance
        effect_size_divided_by_variance_total = np.nansum(
            effect_size_divided_by_variance, axis=0)

        # Calculate the total weight as the inverse of the variance of the effect
        # estimate.
        one_divided_by_variance = 1 / variance
        one_divided_by_variance_total = np.nansum(one_divided_by_variance, axis=0)

        # The total weighted effect size should be divided still by the
        # total weight of each of the studies.
        meta_analysed_effect_sizes = effect_size_divided_by_variance_total / one_divided_by_variance_total
        meta_analysed_standard_error = np.sqrt(1 / one_divided_by_variance_total)

        return pd.DataFrame(np.array([meta_analysed_effect_sizes, meta_analysed_standard_error]).T,
                            index=pd.MultiIndex.from_tuples(combinations), columns=["beta", "standard_error"])


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # We want to assign variant names so that genotype data can be properly written.
    variants_data_frame = sample_variants_from_ref()

    base_output_path = "resources/missingness/"
    cohort_format = "test"

    test_data_generator = TestDataGenerator(
        variants_data_frame, DEFAULT_NUMBER_OF_PHENOTYPES, base_output_path, cohort_format)
    test_data_generator.generate_test_data()

    # Generate
    variant_indices = test_data_generator.generate_variant_indices()
    phenotype_indices = test_data_generator.generate_phenotype_indices()

    # Now design a matrix that introduces missingness
    test_data_generator.write_genotype_data(variant_indices)
    test_data_generator.write_phenotype_data(phenotype_indices)
    test_data_generator.write_covariate_data()
    test_data_generator.write_expected_meta_analysis(
        variant_indices, phenotype_indices
    )

    variant_indices.to_csv("resources/missingness/variant_indices.txt",
                           sep="\t", index=True, index_label="cohort",
                           header=True)

    phenotype_indices.to_csv("resources/missingness/phenotype_indices.txt",
                             sep="\t", index=True, index_label="cohort",
                             header=True)

    # We now want to split the generated data into a number of cohorts,
    # and introduce missingness
    # test_data_generator.


if __name__ == "__main__":
    sys.exit(main())
