#!/usr/bin/env python3

"""
Test script for the classic meta analysis
"""
import glob
import os
import tempfile
import unittest
import numpy as np

# TODO: implement test for missingness in phenotype thingy
# TODO: implement test for missingness in genotype thingy (new getter)
import pandas as pd

from hase import load_mapper
from hdgwas import hdregression, meta_classic
from hdgwas.data import Reader, MetaPhenotype, MetaParData
from hdgwas.tools import merge_genotype, get_intersecting_individual_indices
from unittests.test_hdregression import get_single_study_data
from unittests.utils.hase_executor import HaseExecutor


def load_expected_variant_indices(path):
    return pd.read_csv(path, sep="\t", header=0, index_col="cohort")


class TestClassicMetaAnalyser(unittest.TestCase):

    def setUp(self):
        """
        Method that sets up testing data for this test case.

        Initial steps:
        Use generated data. Load this as if a hase run is started

        For testing the classic meta analyser
        1. Get a mapper ready,
        2. Get a meta phen ready
        3. Get a meta pard ready

        Make sure to set the hase chunking for both variants and genotypes
        to 10 to 20 or so to make sure chunking works as expected

        """
        #self.test_dir = tempfile.mkdtemp()
        self.resources_dir = os.path.join(os.path.dirname(__file__), "resources")
        self.project_root = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = os.path.join(self.resources_dir, "missingness")

        self.test_dir = os.path.join(self.project_root, "unittests", "missingness", "out")
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        self.expected_variant_indices = load_expected_variant_indices(
            os.path.join(self.data_dir, "variant_indices.txt"))

        self.hase_executor = HaseExecutor(self.test_dir, self.project_root,
                                          self.expected_variant_indices.index)

        for cohort_name in self.hase_executor.study_names:
            if not self.hase_executor.per_cohort_results_exist(cohort_name):
                self.hase_executor.per_cohort_analysis(self.data_dir, cohort_name)

        if not os.path.exists(self.hase_executor.get_meta_mapper_folder()):
            self.hase_executor.prepare_mapper_meta_folder()

        if not os.path.exists(self.hase_executor.results_directory):
            self.hase_executor.meta_classic()

    def tearDown(self):
        pass


    def test_mapperGetNonMissing(self):
        """
        """

        expected_genotypes = self.expected_genotypes()

        mapper = load_mapper(
            mapper_chunk_size=30, study_name=self.hase_executor.study_names,
            ref_name=self.hase_executor.ref_name,
            mapper_folder=self.hase_executor.get_meta_mapper_folder(),
            encoded=[0] * len(self.hase_executor.study_names),
            cluster=None, node=None,
            snp_id_inc=None, snp_id_exc=None)

        gen = []
        for i, study_name in enumerate(self.hase_executor.study_names):
            gen.append(Reader('genotype'))
            gen[i].start(os.path.join(self.data_dir, study_name, "genotypes_hdf5"),
                         hdf5=True, study_name=study_name, ID=False)

        keys_list = list()
        actual_genotypes_list = list()

        while True:
            variant_indices, keys = mapper.get(
                allow_missingness=False)

            if isinstance(variant_indices, type(None)):
                break

            keys_list.append(keys)

            actual_genotypes_list.append(merge_genotype(gen, variant_indices, mapper))

        actual_genotypes = np.concatenate(actual_genotypes_list)
        actual_keys = np.concatenate(keys_list)

        self.assertTrue(np.array_equal(
            actual_genotypes,
            expected_genotypes.loc[actual_keys,:].values))


    def test_mapperGetMissingness(self):
        """
        Compare expected variant_indices,
        as created when generating testing data,
        with the actual variant_indices as returned by the
        mapper.get(allow_missingness=T) method.

        Make sure the
        """

        expected_genotypes = self.expected_genotypes()

        mapper = load_mapper(
            mapper_chunk_size=30, study_name=self.hase_executor.study_names,
            ref_name=self.hase_executor.ref_name,
            mapper_folder=self.hase_executor.get_meta_mapper_folder(),
            encoded=[0] * len(self.hase_executor.study_names),
            cluster=None, node=None,
            snp_id_inc=None, snp_id_exc=None)

        gen = []
        for i, study_name in enumerate(self.hase_executor.study_names):
            gen.append(Reader('genotype'))
            gen[i].start(os.path.join(self.data_dir, study_name, "genotypes_hdf5"),
                         hdf5=True, study_name=study_name, ID=False)

        keys_list = list()
        actual_genotypes_list = list()

        while True:
            variant_indices, keys = mapper.get(
                allow_missingness=True)

            if isinstance(variant_indices, type(None)):
                break

            keys_list.append(keys)

            actual_genotypes_list.append(merge_genotype(gen, variant_indices, mapper))

        actual_genotypes = np.concatenate(actual_genotypes_list)
        actual_keys = np.concatenate(keys_list)

        self.assertTrue(np.array_equal(
            actual_genotypes,
            expected_genotypes.loc[actual_keys,:].values))

    def test_CohortAnalyser(self):
        study_name = "test_0"
        study_index = 0
        allow_missingness = True

        genotypes_path = os.path.join(self.data_dir, study_name, "genotypes.csv")
        genotypes = pd.read_csv(genotypes_path, delimiter="\t", index_col='ID')

        phenotypes_path = os.path.join(self.data_dir, study_name, "phenotypes", "test_0.txt")
        phenotypes = pd.read_csv(phenotypes_path, delimiter="\t", index_col='id')

        expected_b4_data_frame = pd.DataFrame(
            hdregression.B4(phenotypes.values, genotypes.values),
            columns=phenotypes.columns, index=genotypes.index)

        expected_results = pd.read_csv(
            os.path.join(self.data_dir, study_name, "results.txt"),
            delimiter="\t")

        expected_betas = expected_results.pivot(index='variant', columns='phenotype', values='Coefficients')

        mapper = load_mapper(
            mapper_chunk_size=200, study_name=self.hase_executor.study_names,
            ref_name=self.hase_executor.ref_name,
            mapper_folder=self.hase_executor.get_meta_mapper_folder(),
            encoded=[1] * len(self.hase_executor.study_names),
            cluster=None, node=None,
            snp_id_inc=None, snp_id_exc=None)

        partial_derivatives_folders = []

        for i, j in enumerate(self.hase_executor.get_folders_all_studies(
                    self.hase_executor.get_partial_derivatives_meta_folder)):
                partial_derivatives_folders.append(Reader('partial'))
                partial_derivatives_folders[i].start(j, study_name=self.hase_executor.study_names[i])
                partial_derivatives_folders[i].folder.load()

        meta_pard = MetaParData(partial_derivatives_folders, self.hase_executor.study_names,
                                protocol=None,
                                allow_missingness=allow_missingness)

        phen = []
        for i, j in enumerate(self.hase_executor.get_folders_all_studies(
                self.hase_executor.get_meta_phenotype_folder)):
            phen.append(Reader('phenotype'))
            phen[i].start(j)

        meta_phen = MetaPhenotype(phen,
            include=None, exclude=None, allow_missingness=allow_missingness)

        gen = []
        for i, j in enumerate(self.hase_executor.get_folders_all_studies(
                    self.hase_executor.get_meta_genotype_folder)):
            gen.append(Reader('genotype'))
            gen[i].start(j, hdf5=True, study_name=self.hase_executor.study_names[i], ID=False)

        # Create a dictionary with the datasets for obtaining
        # the indices of shared identifiers
        datasets = {"phenotype": tuple(i.folder._data for i in phen),
                    "genotype": tuple(i.folder._data for i in gen),
                    "partial_derivatives": tuple(i.folder._data.metadata for i in partial_derivatives_folders)}

        # Get common ids
        row_index, intersecting_identifiers = get_intersecting_individual_indices(datasets)

        classic_meta_analyser = meta_classic.ClassicMetaAnalyser(
            meta_phen, meta_pard, intersecting_identifiers, row_index, self.hase_executor.study_names,
            self.hase_executor.results_directory)

        variant_indices, keys = mapper.get(
            allow_missingness=allow_missingness)
        genotype = merge_genotype(gen, variant_indices, mapper)

        classic_meta_analyser.prepare_cohorts(
            variant_indices, keys)

        phenotype, phenotype_names = classic_meta_analyser.meta_phen.get()
        phenotype_indices = classic_meta_analyser.meta_phen.get_phenotype_indices(phenotype_names)

        cohort = classic_meta_analyser.cohort_list[study_index]
        cohort.set_phenotype_names(phenotype_names)
        cohort.set_phenotype_indices(phenotype_indices[cohort.study_index])
        actual_b4_data_frame = pd.DataFrame(
            cohort.calculate_b4_matrix(genotype, phenotype),
            columns=cohort.get_phenotype_names(phenotype_names),
            index=cohort.get_variant_names(keys))

        self.assertTrue(np.allclose(
            expected_b4_data_frame.loc[
                actual_b4_data_frame.index,
                actual_b4_data_frame.columns].values,
            actual_b4_data_frame.values))

        cohort.analyse(genotype, phenotype)
        actual_betas = pd.DataFrame(
            cohort.get_betas(),
            columns=cohort.get_phenotype_names(phenotype_names),
            index=cohort.get_variant_names(keys))

        self.assertTrue(np.allclose(
            expected_betas.loc[
                actual_betas.index,
                actual_betas.columns].values,
            actual_betas.values))

    def test_MetaPhenotypeGetMissingness(self):
        expected_phenotypes = self.expected_phenotypes()

        meta_phen = self.setup_meta_phen()

        phenotype, phenotype_names = meta_phen.get()

        self.assertTrue(np.array_equal(
            expected_phenotypes.loc[:,phenotype_names].values,
            phenotype))

    def setup_meta_phen(self):
        phen = []
        for i, study_name in enumerate(self.hase_executor.study_names):
            phen.append(Reader('phenotype'))
            phen[i].start(os.path.join(self.data_dir, study_name, "phenotypes"))
        meta_phen = MetaPhenotype(
            phen,
            include=None, exclude=None, allow_missingness=True)
        return meta_phen

    def expected_genotypes(self):
        expected_genotypes_list = list()
        for i, study_name in enumerate(self.hase_executor.study_names):
            expected_genotypes_path = os.path.join(self.data_dir, study_name, "genotypes.csv")
            expected_genotypes_list.append(pd.read_csv(
                expected_genotypes_path, delimiter="\t", index_col='ID'))
        expected_genotypes = pd.concat(expected_genotypes_list, axis=1, sort=True)
        expected_genotypes = expected_genotypes.fillna(0)

        return expected_genotypes

    def test_metaAnalysisResults(self):

        # Load actual results
        file_names = glob.glob(os.path.join(self.hase_executor.results_directory, "*.pkl"))
        actual_results = pd.concat([pd.read_pickle(f) for f in file_names])
        expected_results = pd.read_csv(
            os.path.join(self.data_dir, "meta.txt"), delimiter='\t')
        print(actual_results)

        merged = pd.merge(actual_results,expected_results, how="inner", on=["variant", "phenotype"])
        self.assertTrue(np.allclose(merged.standard_error_x.values, merged.standard_error_y.values))

    def expected_phenotypes(self):
        expected_phenotype_list = list()
        for i, study_name in enumerate(self.hase_executor.study_names):
            expected_phenotypes_path = os.path.join(self.data_dir, study_name, "phenotypes", study_name + ".txt")
            expected_phenotype_list.append(pd.read_csv(
                expected_phenotypes_path, delimiter="\t", index_col='id'))
        expected_phenotypes = pd.concat(expected_phenotype_list, axis=0, sort=True)
        expected_phenotypes = expected_phenotypes.fillna(0)

        return expected_phenotypes


if __name__ == '__main__':
    unittest.main()
