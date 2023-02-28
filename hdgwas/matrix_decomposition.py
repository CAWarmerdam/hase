from __future__ import print_function

from memory_profiler import profile
from sklearn.decomposition import IncrementalPCA

from hdgwas.fake import Encoder
from hdgwas.hdregression import HASE, A_covariates, A_tests, B_covariates, C_matrix, A_inverse, B4, \
    calculate_variant_dependent_a
from hdgwas.tools import study_indexes, Timer
import numpy as np
import os
import time


class MatrixDecomposer(Encoder):
    """
    Decompose matrix to
    """
    def __init__(self, out):
        super(MatrixDecomposer, self).__init__(out)
        self.principal_components = None

    @profile
    def decompose_matrix(self, phen=None, gen=None,
                         mapper=None):

        # Determine intersections
        row_index, intersecting_identifiers = study_indexes(phenotype=phen.folder._data, genotype=gen.folder._data)

        # Phenotype files
        if len([f for f in phen.folder.files if f != 'info_dic.npy']) > 1:
            print('more chunks in phenotype data than expected!')
            raise ValueError("Requiring all phenotypes to be in a single chunk!")

        # Get Phenotype data
        phen.folder.processed = 0
        phenotypes = phen.get_next(index=row_index[1])

        print(phenotypes.shape)
        # Get fitted PCA object
        self.principal_components = IncrementalPCA(n_components=phenotypes.shape[0])

        # Fit each of the genotype chunks.
        self.fit_incrementally(gen, phenotypes, mapper, row_index)

        # Write the eigenvectors and means
        self.save_npy(self.principal_components.components_,
                      save_path=os.path.join('encode_phenotype'),
                      info=phen.folder, index=row_index[1])

        # Export info dict for phenotype axis
        np.save(
            os.path.join(os.path.join(self.out, 'encode_phenotype', 'info_dic.npy')),
            self.phen_info_dic)

        np.save(
            os.path.join(os.path.join(self.out, 'encode_phenotype', 'means.npy')),
            self.principal_components.mean_)

        np.save(
            os.path.join(os.path.join(self.out, 'encode_phenotype', 'explained_variance.npy')),
            self.principal_components.explained_variance_)

        np.save(
            os.path.join(os.path.join(self.out, 'encode_phenotype', 'explained_variance_ratio.npy')),
            self.principal_components.explained_variance_ratio_)

        # Project to PC space
        self.project_components_incrementally(gen, phenotypes, mapper, row_index)

    def fit_incrementally(self, gen, phenotypes, mapper, row_index):
        N_snps_read = 0
        while True:
            with Timer() as t_gen:
                genotype = gen.get_next()
                # Save data when no more genotype chunks are to be iterated over.
                if isinstance(genotype, type(None)):
                    return
                flip = mapper.flip[mapper.folder.name][N_snps_read:N_snps_read + genotype.shape[0]]
                N_snps_read += genotype.shape[0]
                flip_index = (flip == -1)
                genotype = np.apply_along_axis(lambda x: flip * (x - 2 * flip_index), 0, genotype)
                genotype = genotype[:, row_index[0]]

                self.principal_components.partial_fit(B4(phenotypes, genotype))
                print('Time to partially fit genotype {} is {} s'.format(genotype.shape, t_gen.secs))

    def project_components_incrementally(self, gen, phenotypes, mapper, row_index):
        N_snps_read = 0
        while True:
            with Timer() as t_gen:
                genotype = gen.get_next()
                # Save data when no more genotype chunks are to be iterated over.
                if isinstance(genotype, type(None)):
                    return
                flip = mapper.flip[mapper.folder.name][N_snps_read:N_snps_read + genotype.shape[0]]
                N_snps_read += genotype.shape[0]
                flip_index = (flip == -1)
                genotype = np.apply_along_axis(lambda x: flip * (x - 2 * flip_index), 0, genotype)
                genotype = genotype[:, row_index[0]]

                projected_components = self.principal_components.transform(B4(phenotypes, genotype))
                self.save_hdf5(projected_components, os.path.join('encode_genotype'), info=gen.folder,
                            index=row_index[0])
                print('Time to project components {} is {} s'.format(projected_components.shape, t_gen.secs))