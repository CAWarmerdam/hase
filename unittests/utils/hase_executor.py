#!/usr/bin/env python3
import glob
import os
import shutil
import distutils.core
import distutils.dir_util
import sys
import time

import pandas as pd

import hase
from tools import mapper


class HaseExecutor:
    def __init__(self, working_dir, project_root, study_names):

        self.nodes_n = None
        self.node_i = None
        self._covariate_indices_path = None
        self.study_names = study_names
        self.project_root = project_root
        self.test_dir = working_dir
        self.ref_name = "1000Gp1v3_ref"
        self.results_directory = os.path.join(self.test_dir, "results")

        assert os.path.isfile(
            os.path.join(self.project_root, "data", "1000Gp1v3.ref.gz"))
        assert os.path.isfile(
            os.path.join(self.project_root, "data", "1000Gp1v3.ref_info.h5"))

    def set_test_dir(self, working_dir):
        self.test_dir = working_dir

    def get_meta_folder(self, cohort):
        return os.path.join(self.test_dir, "meta_" + cohort)

    def get_partial_derivatives_meta_folder(self, cohort):
        return os.path.join(self.get_meta_folder(cohort), "partial_derivatives")

    def get_meta_genotype_folder(self, cohort):
        return os.path.join(self.get_meta_folder(cohort), "genotype_encoded")

    def get_meta_phenotype_folder(self, cohort):
        return os.path.join(self.get_meta_folder(cohort), "phenotype_encoded")

    def get_mapper_folder(self, cohort):
        return os.path.join(self.test_dir, "mapper_" + cohort, "")

    def per_cohort_analysis(self, data_path, study_name,
                            phenotype_matrix=None):
        # Reference for mapper has to be downloaded

        # Start by running mapper?
        hdf5_genotype_directory = os.path.join(data_path, study_name, "genotypes_hdf5")
        self.per_cohort_mapper_step(hdf5_genotype_directory, study_name)

        # Encode the genotype and phenotype files
        encoding_output = os.path.join(self.test_dir, "encoded")

        if phenotype_matrix is None:
            phenotype_matrix = os.path.join(data_path, study_name, "phenotypes")

        cov_interaction_matrix = os.path.join(data_path, study_name, "covariates")
        self.per_cohort_encoding_step(encoding_output, hdf5_genotype_directory, phenotype_matrix, study_name)

        # Calculate partial derivatives
        partial_derivatives = os.path.join(self.test_dir, "pd")
        self.per_cohort_partial_derivatives(cov_interaction_matrix,
                                            hdf5_genotype_directory, partial_derivatives,
                                            phenotype_matrix, study_name)

        self.finalize_per_cohort(encoding_output, hdf5_genotype_directory, partial_derivatives, study_name)

    def per_cohort_mapper_step(self, hdf5_genotype_directory, study_name):
        if not os.path.exists(self.get_mapper_folder(study_name)):
            mapper.main(["-g", hdf5_genotype_directory,
                         "-study_name", study_name,
                         "-ref_name", self.ref_name,
                         "-o", self.get_mapper_folder(study_name)])

    def per_cohort_partial_derivatives(self, cov_interaction_matrix, hdf5_genotype_directory, partial_derivatives,
                                       phenotype_matrix, study_name):
        hase.main(["-g", hdf5_genotype_directory,
                   "-study_name", study_name,
                   "-o", partial_derivatives,
                   "-mapper", self.get_mapper_folder(study_name),
                   "-ph", phenotype_matrix,
                   "-cov", cov_interaction_matrix,
                   "-ref_name", self.ref_name,
                   "-mode", "single-meta"])

    def finalize_per_cohort(self, encoding_output, hdf5_genotype_directory, partial_derivatives, study_name):
        # Create directory structure required for meta-analysis
        # Define directory names.
        phenotype_meta = self.get_meta_phenotype_folder(study_name)
        individuals_meta = os.path.join(self.get_meta_genotype_folder(study_name), "individuals")
        actual_genotype_meta = os.path.join(self.get_meta_genotype_folder(study_name), "genotype")
        # Copy the probes to genotype_meta. This calls os.makedirs() always,
        # and throws an exception if it already exists.
        # First we remove the genotype_meta if it does already exist
        if os.path.exists(self.get_meta_folder(study_name)):
            shutil.rmtree(self.get_meta_folder(study_name))
        shutil.copytree(os.path.join(hdf5_genotype_directory, "probes"),
                        os.path.join(self.get_meta_genotype_folder(study_name), "probes"))
        # Make the required directories
        # os.makedirs(actual_genotype_meta)
        os.makedirs(individuals_meta)
        os.makedirs(self.get_partial_derivatives_meta_folder(study_name))
        os.makedirs(phenotype_meta)
        # Copy the required data
        distutils.dir_util.copy_tree(os.path.join(encoding_output, "encode_individuals"), individuals_meta)
        distutils.dir_util.copy_tree(os.path.join(encoding_output, "encode_genotype"), actual_genotype_meta)
        distutils.dir_util.copy_tree(os.path.join(encoding_output, "encode_phenotype"), phenotype_meta)
        for file in glob.glob(os.path.join(partial_derivatives, "*.npy")):
            shutil.copy(file, self.get_partial_derivatives_meta_folder(study_name))
        distutils.dir_util.remove_tree(encoding_output)
        distutils.dir_util.remove_tree(partial_derivatives)

    def per_cohort_encoding_step(self, encoding_output, hdf5_genotype_directory, phenotype_matrix, study_name):
        hase.main(["-g", hdf5_genotype_directory,
                   "-study_name", study_name,
                   "-o", encoding_output,
                   "-mapper", self.get_mapper_folder(study_name),
                   "-ph", phenotype_matrix,
                   "-ref_name", self.ref_name,
                   "-mode", "encoding"])

    def prepare_mapper_meta_folder(self):
        meta_mapper = self.get_meta_mapper_folder()
        os.makedirs(meta_mapper)
        for i, study_name in enumerate(self.study_names):
            mapper_cohort = self.get_mapper_folder(study_name)

            shutil.copy(
                os.path.join(mapper_cohort, "".join(["flip_", self.ref_name, "_", study_name, ".npy"])),
                meta_mapper)

            shutil.copy(
                os.path.join(mapper_cohort, "".join(["values_", self.ref_name, "_", study_name, ".npy"])),
                meta_mapper)

            if i == 0:
                shutil.copy(
                    os.path.join(mapper_cohort, "".join(["keys_", self.ref_name, ".npy"])),
                    meta_mapper)

    def meta_classic(self):
        self.run_analysis(mode="meta-classic")

    def run_analysis(self, mode="meta-classic"):
        # We need to get the mapper files. The mapper files 'flip' and 'values' have to be moved to a single
        # folder
        command_of_lists = [["-g"], self.run_func_all_studies(self.get_meta_genotype_folder),
                            ["-study_name"], self.study_names,
                            ["-ph"], self.run_func_all_studies(self.get_meta_phenotype_folder),
                            ["-derivatives"], self.run_func_all_studies(self.get_partial_derivatives_meta_folder),
                            ["-mapper", self.get_meta_mapper_folder()],
                            ["-o", self.results_directory],
                            ["-encoded"], ["1"] * len(self.study_names),
                            ["-ref_name", self.ref_name],
                            ["-snp_id_inc", "/Users/cawarmerdam/Documents/hdaa/hase_repository/unittests/largedataset/out/snp_id_inc.txt"],
                            ["-mapper_chunk", "500"],
                            ["-mode", mode],
                            ["-allow_missingness"],
                            ["-th", "0"],
                            ["-maf", "0.05"]]
        covariate_indices_path = self.get_covariate_indices_path()
        if covariate_indices_path:
            command_of_lists.append(["--covariate-indices", covariate_indices_path])
        if self.nodes_n is not None and self.node_i is not None:
            command_of_lists.append(
                ["-node", self.nodes_n, self.node_i, "-cluster", "y"])
        argv = [str(arg) for sub_arg in command_of_lists for arg in sub_arg]
        hase.main(argv)

    def run_func_all_studies(self, func):
        return [func(cohort_name) for cohort_name in self.study_names]

    def get_meta_mapper_folder(self):
        return os.path.join(self.test_dir, "meta_mapper", "")

    def per_cohort_results_exist(self, study_name):
        path_presence_list = [
            os.path.exists(self.get_meta_genotype_folder(study_name)),
            os.path.exists(self.get_meta_phenotype_folder(study_name)),
            os.path.exists(self.get_partial_derivatives_meta_folder(study_name)),
            os.path.exists(self.get_mapper_folder(study_name))]

        return False not in path_presence_list

    def meta_analyse(self, data_dir):
        for cohort_name in self.study_names:
            if not self.per_cohort_results_exist(cohort_name):
                self.per_cohort_analysis(data_dir, cohort_name)

        if not os.path.exists(self.get_meta_mapper_folder()):
            self.prepare_mapper_meta_folder()

        if not os.path.exists(self.results_directory):
            self.meta_classic()

    def get_covariate_indices_path(self):
        return self._covariate_indices_path

    def set_covariate_indices_path(self, path):
        self._covariate_indices_path = path

    def copy_per_cohort_analysis(self, other):
        for cohort_name in self.study_names:
            meta_folder = other.get_meta_folder(cohort_name)
            if not os.path.exists(meta_folder):
                shutil.copytree(
                    self.get_meta_folder(cohort_name),
                    meta_folder)

    def copy_per_cohort_mapper_files(self, other):
        for cohort_name in self.study_names:
            mapper_folder = other.get_mapper_folder(cohort_name)
            if not os.path.exists(mapper_folder):
                shutil.copytree(
                    self.get_mapper_folder(cohort_name),
                    mapper_folder)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    resources_dir = os.path.join("..", "resources")
    project_root = os.path.join("..", "..")
    dataset = "largedataset"
    data_dir = os.path.join(resources_dir, "%s" % dataset)

    test_dir = os.path.join(project_root, "unittests", dataset, "out")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    study_names = ["test_0", "test_1", "test_2"]

    hase_executor = HaseExecutor(test_dir, project_root,
                                 study_names)
    hase_executor.nodes_n = 2
    hase_executor.node_i = 1

    hase_executor.meta_analyse(data_dir)

    # timestamps_as_datetime = list()
    # # for memory, timestamp in mem_usage:
    # #     timestamps_as_datetime.append(time.gmtime(timestamp))
    #
    # # mem_usage_df[1] = timestamps_as_datetime
    # print(mem_usage_df)
    # print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)


    return 0


if __name__ == "__main__":
    sys.exit(main())
