from __future__ import print_function
import gc
import sys
import os
import warnings

import numpy as np
import pandas as pd
from hdgwas.tools import Timer, Checker, study_indexes, Mapper, HaseAnalyser, merge_genotype, Reference, timing, \
    check_np, check_converter, get_intersecting_individual_indices, select_identifiers
from hdgwas.hdregression import HASE, A_covariates, A_tests, B_covariates, C_matrix, A_inverse, B4, \
    get_a_inverse_extended, hase_supporting_interactions, expand_sample_size_matrix, \
    a_inverse_extended_allow_missingness


class HaseException(Exception):
    pass


class ClassicMetaAnalyser:
    def __init__(self, meta_phen, meta_pard,
                 sample_intersection, sample_indices, study_names, out):
        self.meta_pard = meta_pard
        self.meta_phen = meta_phen
        self.out = out
        self.cohort_list = self.construct_cohort_list(
            meta_pard, sample_intersection, sample_indices, study_names)
        self.output_type = "npy"
        self.results = None
        self.results_index = 0

    @staticmethod
    def construct_cohort_list(meta_pard, sample_intersection,
                              sample_indices, study_names):
        # Build a list of studies / cohorts
        cohort_list = list()
        # Loop through studies / cohort names
        for study_index, study_name in enumerate(study_names):
            # Determine sample size
            sample_size = len(meta_pard.pd[study_name].folder._data.metadata['id'])

            # Now get the row indices matching with this studies samples,
            # so we can extract genotype and phenotype samples for this
            # study only.
            study_row_indices = select_identifiers(
                study_index, sample_indices, sample_intersection)

            # Initialize the cohort analysis
            cohort = CohortAnalyser(
                study_index, study_name, study_row_indices, sample_size)

            # Add a dictionary as an index matching phenotypes to
            # indices in the partial derivatives C and b_cov
            keys = meta_pard.phen_mapper.dic.keys()
            phen_ind_dic = {k: i for i, k in enumerate(keys)}
            cohort.set_partial_derivatives_phenotype_index(phen_ind_dic)
            cohort_list.append(cohort)
        return cohort_list

    def analyse_genotype_chunk(self, genotype, variant_names, variant_indices):

        # First, prepare the genotype data for this chunk, per cohort
        for cohort in self.cohort_list:
            cohort.set_variant_names(variant_names)
            # Get the partial derivatives for each study
            with Timer() as t_pd:
                # Extract the partial derivatives from the data sources
                a_test, b_cov, C, a_cov = self.meta_pard.get_single_study(
                    study_name=cohort.study_name,
                    study_index=cohort.study_index,
                    variant_indices=variant_indices)

            print("Time to get PD {}s".format(t_pd.secs))

            # Add the partial derivatives for this cohort
            cohort.set_partial_derivatives(a_test, b_cov, C, a_cov)
            # Add the maf filter for this cohort
            cohort.set_observed_maf(self.meta_pard.minor_allele_frequencies_study(
                study_name=cohort.study_name,
                study_index=cohort.study_index,
                SNPs_index=variant_indices))
            cohort.set_variant_indices(variant_indices[cohort.study_index])
            cohort.set_variant_filter(cohort.maf_filter())
            # Apply filters on partial derivatives and calculate processed
            # data structures.
            cohort.finalize_partial_derivatives()

        while True:
            # Get the next phenotype chunk.
            with Timer() as t_ph:
                # Phen names is the
                phenotype, phenotype_names = self.meta_phen.get()
                phenotype_indices = self.meta_phen.get_phenotype_indices(phenotype_names)
            print("Time to get PH {}s".format(t_ph.secs))

            # If the phenotype type is None, the loop is done...
            if isinstance(phenotype, type(None)):
                # Reset the processed phenotypes when the loop is done.
                # With the next chunk of SNPs we need to do these again
                self.meta_phen.processed = 0
                # The encoded interactions are also processed at the
                # same rate as the phenotypes.
                # Reset the number of processed values for this as well.
                print('All phenotypes processed!')
                break
            print("Merged phenotype shape {}".format(phenotype.shape))

            for cohort in self.cohort_list:
                # Now we need to do the analysis with the selected
                # phenotypes and genotypes
                cohort.set_phenotype_names(phenotype_names)
                cohort.set_phenotype_indices(phenotype_indices[cohort.study_index])
                cohort.analyse(genotype, phenotype)

            # Now do the classic meta-analysis
            self.meta_analyse(variant_names, phenotype_names)

    def meta_analyse(self, variant_names, phenotype_names):
        """
        Meta analysis for all associations.

        For each association, we apply an inverse variance meta analyis.

        From the analysis of individual cohorts we can extract a number of
        summary statistics in a 3d numpy array:
        -   The 1st dimension of this array comprises each of the included
            phenotypes for the respective cohort or study.
        -   The 2nd dimension of this array does not seem to have a function.
            (could this perhaps have been meant for interaction effects)
        -   The 3rd dimension of this array represents all of the phenotypes
            that are tested for.

        We need to combine each of these dimensions so that variants match
        with each other and we can accurately do a meta-analysis.

        For combining the results between cohorts, we complete the matrices
        with all missing values for ease of use.
        :param variant_names:
        :param phenotype_names:

        """
        # Now we need to do a meta-analysis
        # Lets first find out how we need to code missing associations to not let these bias results.

        # TODO: get a matrix that details what variant gene combination is missing
        # Missing variants at the start are listed in variant_indices
        # We should harmonise the variant indices with the variant_filter

        # The same should be done for the phenotypes

        # For every study, get summary statistics for all combinations that
        # are present.
        effect_sizes = self.get_effect_sizes()
        standard_error = self.get_standard_error()
        variance = standard_error ** 2

        effect_size_divided_by_variance = effect_sizes / variance
        effect_size_divided_by_variance_total = np.nansum(
            effect_size_divided_by_variance, axis=0)

        one_divided_by_variance = 1 / variance
        one_divided_by_variance_total = np.sum(one_divided_by_variance, axis=0)

        meta_analysed_effect_sizes = effect_size_divided_by_variance_total / one_divided_by_variance_total
        meta_analysed_standard_error = np.sqrt(1 / one_divided_by_variance_total)
        meta_analysis_sample_sizes = self.get_sample_sizes()

        self.set_results(meta_analysed_effect_sizes, meta_analysed_standard_error, meta_analysis_sample_sizes,
                         phenotype_names, variant_names)

        # For every cohort, add the effect size and se

        # Return estimate, return se, return rsid, return phenotype

    def set_results(self, meta_analysed_effect_sizes, meta_analysed_standard_error, meta_analysis_sample_sizes,
                    phenotype_names, variant_names):
        
        self.results = pd.DataFrame({
            "variant": np.tile(variant_names, meta_analysed_effect_sizes.shape[1]),
            "phenotype": np.repeat(phenotype_names, meta_analysed_effect_sizes.shape[0]),
            "beta": meta_analysed_effect_sizes.flatten(),
            "standard_error": meta_analysed_standard_error.flatten(),
            "sample_size": meta_analysis_sample_sizes.flatten()})

    def save_results(self, chunk = None, node = None):

        print('Saving results to {}'.format(self.out))

        # Possibly set a threshold on p-value or Z score or t-statistic
        results = self.results

        if chunk is None:
            np.save(os.path.join(self.out, str(self.results_index) + 'result.npy'), results)
        else:
            np.save(os.path.join(self.out, 'node' + str(node) + '_' + str(chunk) + '_' + str(
                chunk) + '_' + str(self.results_index) + 'result.npy'), results)
        self.results_index += 1
        

    def get_effect_sizes(self):
        """
        Method that returns a matrix of effect sizes.
        The first dimension represents variants,
        the second dimension represents the studies / cohorts.

        Missing associations are NA
        :param association_indices: A 3d matrix with for each study an array of variant indices, indicating at what position
        variants are located.
        :return: matrix of effect sizes [assoc, studies]
        """
        return np.stack([cohort.analyser.get_betas() for cohort in self.cohort_list])

    def get_standard_error(self):
        """
        Method that returns a matrix of standard errors of effect sizes.
        The first dimension represents associations,
        the second dimension represents the studies / cohorts.
        :param association_indices: A list with for each study an array of variant indices, indicating at what position
        variants are located
        """
        return np.stack([cohort.analyser.standard_error for cohort in self.cohort_list])

    def get_sample_sizes(self):
        return np.nansum(np.stack([cohort.get_sample_sizes() for cohort in self.cohort_list]), axis=0)

class CohortAnalyser:
    """
    Class that handles HASE analysis for a single cohort.

    Attempts to limit memory usage as much as possible. It is written
    with the following workflow in mind:

    Initialize an instance using basic metadata for the specific
    cohort.

    For each genotype chunk, we can do the following with this class

    1. Process and store partial derivative structures on a genotype level that
    require much time to process. By doing this, we limit having
    to recalculate these for every set of phenotypes.

    2. Do only store filters for variants and phenotypes. Do not store
    actual filtered data. This will accumulate to quite a bit of data
    when looping over phenotypes without clearing.

    3. Do the analysis, filtering data on the fly.


    """

    def __init__(self, study_index, study_name, row_index,
                 sample_size,
                 maf_threshold=0,
                 a_test=None, b_cov=None, C=None, a_cov=None,
                 partial_derivatives_phenotype_index=None):
        self.unknown_dimension_size = 1
        self.partial_derivatives_phenotype_index = partial_derivatives_phenotype_index
        self.sample_size = sample_size
        self.a_cov = a_cov
        self.a_test = a_test
        self.a_inv = None
        self.phenotype_names = None
        self.variant_indices = None
        self.phenotype_indices = None
        self.C = C
        self.b_cov = b_cov
        self.study_name = study_name
        self.study_index = study_index
        self.sample_indices = row_index
        self.analyser = HaseAnalyser()
        self.number_of_variable_terms = None
        self.number_of_constant_terms = None
        self._finalized = False
        self.maf_threshold = maf_threshold

    def set_partial_derivatives(self, a_test, b_cov, C, a_cov):
        """
        Set partial derivatives.
        """
        self.b_cov = b_cov
        self.C = C
        self.a_test = a_test
        self.a_cov = a_cov

    def set_partial_derivatives_phenotype_index(self, index):
        """
        Set partial derivatives phenotype index. :param index: the index should be a dictionary with the names of the
        phenotype as keys, and the accompanying index in the partial derivatives as values.
        """
        self.partial_derivatives_phenotype_index = index

    def finalize_partial_derivatives(self):
        if self.variant_indices is None:
            raise HaseException("variant indices must be set prior to finalizing partial derivatives")
        self.a_test = self.a_test[self.variant_indices != -1]

        # Use new A_inverse that supports variable number of non-constant A parts
        self.a_inv = get_a_inverse_extended(self.a_cov, self.a_test)

        self._set_number_of_variable_terms()
        self._set_number_of_constant_terms()
        self._finalized = True

    def _set_number_of_constant_terms(self):
        self.number_of_constant_terms = self.a_inv.shape[1] - self.a_test.shape[2]

    def _set_number_of_variable_terms(self):
        self.number_of_variable_terms = self.a_test.shape[2]

    def set_observed_maf(self, maf):
        self.analyser.maf = maf

    def maf_filter(self):
        threshold = self.get_maf_threshold()
        variant_filter = [True] * self.analyser.maf.shape[0]
        if threshold != 0:
            variant_filter = (self.analyser.maf > threshold) \
                             & (self.analyser.maf < 1 - threshold) \
                             & (self.analyser.maf != 0.5)
            # This probably handles the large a_test matrices as well.
        return variant_filter

    def set_variant_indices(self, variant_indices):
        self.variant_indices = variant_indices

    def set_variant_filter(self, variant_filter):
        self.variant_indices[not variant_filter] = -1

    def set_phenotype_names(self, phenotype_names):
        self.phenotype_names = phenotype_names

    def set_phenotype_indices(self, phenotype_indices):
        self.phenotype_indices = phenotype_indices

    def apply_variant_filters(self, genotype):
        variant_indices = self.variant_indices
        if variant_indices is None:
            variant_filter = [True] * genotype.shape[0]
        else:
            variant_filter = self.variant_indices != -1

        filtered_genotype = genotype[np.ix_(variant_filter, self.sample_indices['genotype'])]

        if genotype.shape[0] == 0:
            raise HaseException("No SNPs pass variant filters")

        return filtered_genotype

    def apply_phenotype_filters_on_phenotype_matrix(self, phenotype):
        phenotype_indices = self.phenotype_indices
        if phenotype_indices is None:
            phenotype_filter = [True] * phenotype.shape[1]
        else:
            phenotype_filter = self.phenotype_indices != -1

        filtered_phenotype = phenotype[np.ix_(self.sample_indices['phenotype'], phenotype_filter)]

        print("Selected phenotype shape {}".format(filtered_phenotype.shape))

        return filtered_phenotype

    def get_a_inverse(self):
        return self.a_inv

    def get_b_cov(self):
        """
        This method returns the covariable part of b.
        :return: matrix b_cov
        """
        phenotype_indexes_for_partial_derivatives = self.get_phenotype_slicer_for_partial_derivatives()
        return self.b_cov[:, phenotype_indexes_for_partial_derivatives]

    def get_C(self):
        """
        This method returns the C matrix
        :return: matrix C as a 2d numpy array.
        """
        phenotype_indexes_for_partial_derivatives = self.get_phenotype_slicer_for_partial_derivatives()
        return self.C[phenotype_indexes_for_partial_derivatives]

    def get_phenotype_slicer_for_partial_derivatives(self):
        """
        Method that returns an array of indexes with which phenotypes can
        be selected from the partial derivatives.

        We first need to apply the phenotype filters to only select those
        phenotypes that we wish to use in this cohort / analysis.

        Then we use this filtering stuff and the partial derivatives phenotype
        index to get the indices that match the phenotypes.

        :return: Array of indices.
        """
        # First make a local copy of the phenotype filter
        phenotype_indices = self.phenotype_indices
        # Then, check if the phenotype filter has been set, if this is not the case,
        # overwrite the filter with True. This selects every phenotype
        if phenotype_indices is None:
            phenotype_filter = [True] * self.phenotype_names.shape[0]
        else:
            phenotype_filter = self.phenotype_indices != -1

        # Now, using the filter, select all names that we want to select
        phenotype_names = self.phenotype_names[phenotype_filter]
        # Now we match the filter phenotype names to individual indexes.
        phenotype_indexes_for_partial_derivatives = np.array(
            [self.partial_derivatives_phenotype_index[name] for name in phenotype_names])
        if len(phenotype_indexes_for_partial_derivatives) == 0:
            raise HaseException('There is no common ids in phenotype files and PD data!')
        else:
            print('There are {} common ids in phenotype files and PD data!'.format(
                len(phenotype_indexes_for_partial_derivatives)))
        return phenotype_indexes_for_partial_derivatives

    def analyse(self, genotype, phenotype):

        # Get partial derivatives
        C_test = self.get_C()
        b_cov_test = self.get_b_cov()
        a_inverse = self.get_a_inverse()

        # Determine the number of constant terms
        number_of_constant_terms = self.get_number_of_constant_terms()

        # Calculate the B4 matrix
        b4 = B4(self.apply_phenotype_filters_on_phenotype_matrix(phenotype),
                self.apply_variant_filters(genotype))
        b_variable = b4[np.newaxis, ...]

        # Calculate the degrees of freedom
        degrees_of_freedom = (self.sample_size - a_inverse.shape[1])

        print("B4 shape is {}".format(b4.shape))
        t_stat, standard_error = hase_supporting_interactions(
            b_variable, a_inverse, b_cov_test, C_test,
            number_of_constant_terms, degrees_of_freedom)

        # TODO: test that missing values and filtered out values are not present in the input and the output.

        self.analyser.t_stat = self.complete_output_with_missingness(t_stat[:, 0, :])
        self.analyser.standard_error = self.complete_output_with_missingness(standard_error[:, 0, :])
        # Now we have to store the t-stat, and standard_error in a data structure from
        # which we can do the meta analysis.

    def complete_output_with_missingness(self, output_without_missingness):
        output_with_missingness = np.full((self.variant_indices.shape[0], self.phenotype_indices.shape[0]), np.NaN)
        output_with_missingness[np.ix_(self.variant_indices != -1, self.phenotype_indices != -1)] = output_without_missingness
        return output_with_missingness

    def get_number_of_constant_terms(self):
        return self.number_of_constant_terms

    def get_maf_threshold(self):
        return self.maf_threshold

    def set_variant_names(self, variant_names):
        self.analyser.rsid = variant_names

    def get_sample_sizes(self):
        sample_sizes = np.full((self.variant_indices.shape[0], self.phenotype_indices.shape[0]), self.sample_size)
        sample_sizes[np.ix_(self.variant_indices == -1, self.phenotype_indices == -1)] = 0
        return sample_sizes
