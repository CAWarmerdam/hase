from __future__ import print_function

import argparse
import sys
from itertools import chain

import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt
from scipy.stats import genpareto
from scipy.stats import chisquare
from scipy.stats import norm
from scipy.stats import gaussian_kde
from sklearn.utils import stats

sns.set_style(style="ticks")

def get_permuted_p_value(
        distribution, n_exceedances, n_permuted_statistics, test_statistic):
    return ((n_exceedances / n_permuted_statistics)
            * (1-(distribution.pdf(np.abs(test_statistic)))))


def get_permutation_pvalues(
        empirical_dataframe, permutation_dataframe, permutation_distributions,
        maf, test, perm):

    # Also define a dictionary to store row boolean indices for each maf bin
    permutation_pvalues_per_maf_bin = list()

    # Now loop through the maf bins to calculate M and store maf indices
    for maf_bin in permutation_dataframe[maf].sort_values().unique():

        # Get the permutation distributions
        permutation_distribution = permutation_distributions[maf_bin][0]
        permutation_exceedances = permutation_distributions[maf_bin][1]

        # Get the permuted dataframe for this maf bin
        permuted_dataframe_bin = (
            permutation_dataframe[permutation_dataframe[maf] == maf_bin]
            .sort_values(by=[perm])
            .drop([maf], axis=1))

        # Exceedances threshold
        exceedances_threshold = (
            permuted_dataframe_bin[perm].iloc[-permutation_exceedances]
            + permuted_dataframe_bin[perm].iloc[-(permutation_exceedances + 1)]) / 2

        # Get the number of permuted values
        n_permuted = float(permuted_dataframe_bin.shape[0])

        # Compute M
        permuted_dataframe_bin['M'] = (
            permuted_dataframe_bin[perm].rank(ascending=False))

        empirical_dataframe_bin = (
            empirical_dataframe[empirical_dataframe[maf] == maf_bin]
            .sort_values(by=[test])).drop([perm], axis=1)

        print(np.sum(empirical_dataframe[maf] == maf_bin))

        pvalue_dataframe_bin = pd.merge_asof(
            empirical_dataframe_bin,
            permuted_dataframe_bin,
            left_on=test, right_on=perm,
            direction="forward"
        )

        pvalue_dataframe_bin.loc[pvalue_dataframe_bin['M'].isna(), 'M'] = 0

        print(pvalue_dataframe_bin[np.array([test, perm, 'M'])])

        print(maf_bin)
        print(pvalue_dataframe_bin.shape[0])

        # Compute P'ecdf for all empirical results
        pvalue_dataframe_bin['P_ecdf'] = (
                pvalue_dataframe_bin['M'] / n_permuted)

        pvalue_dataframe_bin['P_ecdf_pseudocount'] = (
                (pvalue_dataframe_bin['M'] + 1) / n_permuted)

        # For those empirical results that have an M < 10
        pvalue_dataframe_bin['P_gpd'] = -1

        # Select rows for which we can use the GPD
        gpd_rows = (pvalue_dataframe_bin['M'] < 10)

        # Get the generalized pareto distribution pdf for the rows
        # that are applicable for the GPD
        gpd_cdf = permutation_distribution.cdf(
            pvalue_dataframe_bin.loc[gpd_rows, test] - exceedances_threshold)

        print(permutation_exceedances)
        print(n_permuted)
        print(permutation_exceedances / n_permuted)

        # calculate the accurate permuted p-value
        pvalue_dataframe_bin.loc[gpd_rows, 'P_gpd'] = (
            permutation_exceedances / float(n_permuted) * (1 - gpd_cdf))

        pvalue_dataframe_bin['P_accurate'] = (
            pvalue_dataframe_bin['P_ecdf']
        )

        pvalue_dataframe_bin.loc[gpd_rows, 'P_accurate'] = (
            pvalue_dataframe_bin.loc[gpd_rows, 'P_gpd']
        )

        permutation_pvalues_per_maf_bin.append(pvalue_dataframe_bin)

    return pd.concat(permutation_pvalues_per_maf_bin)


def anderson_darling_gpd(sorted_values, genpareto_dist, genpareto_shape, table):
    N = len(sorted_values)

    # Get the cumulative density function, log transformed
    logcdf = genpareto_dist.logcdf(sorted_values)
    # Get the survival function, log transformed
    logsf = genpareto_dist.logsf(sorted_values)

    i = np.arange(1, N + 1)
    anderson_darling = (
            -N - np.sum((2*i - 1.0) / N * (logcdf + logsf[::-1]), axis=0))

    significance_values_character = np.array(table.columns[1:table.shape[1]])
    critical_values = np.array([np.interp(genpareto_shape, table["k"], table[col])
                                for col in significance_values_character])

    significance_values = significance_values_character.astype(float)

    return anderson_darling, critical_values, significance_values


def _fit_distribution(abs_standard_scores_ordered, n_exceedances, anderson_darling_table):
    # Calculate the threshold value
    exceedances_threshold = (
        abs_standard_scores_ordered[n_exceedances-1]
        + abs_standard_scores_ordered[n_exceedances]) / 2

    # Calculate the z argument for the generalized pareto distribution
    exceedence_values_sorted_ascending = np.sort(
        abs_standard_scores_ordered[(abs_standard_scores_ordered > exceedances_threshold)])
    z_test_statistics_for_fit = (
            exceedence_values_sorted_ascending
            - exceedances_threshold)

    # Fit the generalized pareto distribution
    fitted_parameters = genpareto.fit(z_test_statistics_for_fit)

    # Generate the probability density function
    genpareto_dist = genpareto(c=fitted_parameters[0],
                               loc=fitted_parameters[1],
                               scale=fitted_parameters[2])

    ad_statistic, crit_values, pvalues = anderson_darling_gpd(
        z_test_statistics_for_fit, genpareto_dist,
        fitted_parameters[0], anderson_darling_table)

    goodness_of_fit_pval = (pvalues[ad_statistic >= crit_values]).min(initial=1)

    if goodness_of_fit_pval <= 0.05 and n_exceedances >= 10:
        return _fit_distribution(
            abs_standard_scores_ordered, n_exceedances-10, anderson_darling_table)

    elif goodness_of_fit_pval > 0.05:
        print("Goodness of fit pvalue:")
        print(goodness_of_fit_pval)
        return genpareto_dist, n_exceedances

    else:
        return None, 0


def plot_genpareto(standard_scores, genpareto_fit, n_exceedances, label):
    abs_standard_scores_ordered = np.sort(np.abs(standard_scores))[::-1]

    # Calculate the threshold value
    exceedances_threshold = (
        abs_standard_scores_ordered[n_exceedances]
        + abs_standard_scores_ordered[n_exceedances+1]) / 2

    # Calculate the z argument for the generalized pareto distribution
    exceedence_values_sorted_ascending = np.sort(
        abs_standard_scores_ordered[(abs_standard_scores_ordered > exceedances_threshold)])
    z_test_statistics_for_fit = (
            exceedence_values_sorted_ascending
            - exceedances_threshold)

    # Calculate empirical CDF
    empirical_cdf = (
            np.arange(1, len(exceedence_values_sorted_ascending)+1)
            / float(len(exceedence_values_sorted_ascending)))

    # X values
    x_values = np.arange(z_test_statistics_for_fit.min(),
                         z_test_statistics_for_fit.max(), 0.1)
    theoretical_cdf = genpareto_fit.cdf(x_values)

    sns.lineplot(x=exceedence_values_sorted_ascending, y=empirical_cdf, label="empirical CDF")
    threshold = x_values + exceedances_threshold
    sns.lineplot(x=threshold, y=theoretical_cdf, label="theoretical CDF")
    sns.rugplot(a=exceedence_values_sorted_ascending)
    plt.text(exceedances_threshold, 1, label, horizontalalignment='left', size='medium', color='black',
             weight='semibold')
    plt.legend()
    plt.show()


def fit_distribution(
        standard_scores, anderson_darling_table, n_exceedances_init=250):
    # Compute the permutation p-value using the
    # P_gpd method
    abs_standard_scores_ordered = np.sort(np.abs(standard_scores))[::-1]
    return _fit_distribution(
        abs_standard_scores_ordered,
        n_exceedances_init,
        anderson_darling_table)


def plot_pdf_over_maf(genpareto_distributions, statistics, maf_bins):
    # Every genpareto distribution starts at a different threshold
    x_values = np.arange(0,
                         np.concatenate(statistics.values()).max(), 0.1)
    dataframe_dict = dict()

    for maf_bin, (genpareto_fit, n_exceedances) in genpareto_distributions.items():
        # First calculate the exceedances threshold.
        abs_standard_scores_ordered = np.sort(np.abs(statistics[maf_bin]))[::-1]
        print(maf_bin)
        print(abs_standard_scores_ordered[n_exceedances - 1])
        print(abs_standard_scores_ordered[n_exceedances])

        # Calculate the threshold value
        exceedances_threshold = (
            abs_standard_scores_ordered[n_exceedances - 1]
            + abs_standard_scores_ordered[n_exceedances]) / 2

        density_function = genpareto_fit.pdf(x_values - exceedances_threshold)
        dataframe_dict[maf_bin] = pd.DataFrame({
            "x": x_values,
            "y": density_function})

    concatenated = pd.concat(
        dataframe_dict.values(),
        keys=dataframe_dict.keys()).rename_axis(['maf', 'i']).reset_index()

    concatenated["maf"] = pd.Categorical(concatenated["maf"],
                                         categories=maf_bins, ordered=True)

    ax = sns.lineplot(x='x', y='y', hue='maf', data=concatenated,
                      palette="viridis", hue_order=maf_bins)
    ax.set(ylim=(0, 0.1))
    plt.show()


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(description="Correct p-values")
    parser.add_argument("--AD-test-table", dest="anderson_darling_path",
                        help="Anderson-Darling table for determining "
                             "goodness-of-fit p-values of generalized "
                             "pareto distributions")
    parser.add_argument("--emp",
                        help="File with empirical association statistics")
    parser.add_argument("--perm",
                        help="File with permutation association statistics")
    parser.add_argument("--breakpoints",
                        help="File with breakpoints for maf bins")
    parser.add_argument("--variants",
                        help="File with uncorrelated variants per maf bin")

    # Load the anderson-darling table
    anderson_darling_table = (
        pd.read_csv("anderson_darling_table.csv").iloc[1::2, :])

    # Load data
    assoc = pd.read_csv("/Users/cawarmerdam/Downloads/PRPF8_betas.txt.gz", sep="\t")
    assoc["Z_emp_abs"] = (assoc["Beta_emp"] / assoc["Se_emp"]).abs().pow(3)
    assoc["Z_perm_abs"] = (assoc["Beta_perm"] / assoc["Se_perm"]).abs().pow(3)

    assoc["perm_log10_pval"] = -np.log10(assoc["Pperm"])

    # Check for NAs etc.
    print(assoc["Pperm"].isna().any())
    print((assoc["Pperm"]==0).any())

    sns.scatterplot(x="MAF", y="Z_perm_abs", marker="+",
                    data=assoc)
    plt.show()

    uncorrelated_variants = pd.read_csv(
        "/Users/cawarmerdam/Documents/projects/eQTLGen/"
        "EstBB_testing/CorrectEmpericalResults/GetUncorrelatedSnps/"
        "out_uncorrelated_snps_per_maf_bin_2.txt", sep=" ")

    breakpoints = np.insert(
        pd.read_csv("/Users/cawarmerdam/Documents/projects/eQTLGen/"
        "EstBB_testing/CorrectEmpericalResults/data/"
        "breakpoints.txt", header=None)[0].values, 0,0)
    maf_bins = ["{0:.3g}-{1:.3g}".format(
        breakpoints[i], breakpoints[i+1])
        for i in range(len(breakpoints) - 1)]

    assoc["MAF_BIN"] = pd.cut(
        assoc["MAF"], breakpoints,
        right=True, include_lowest=True,
        labels=maf_bins)

    uncorrelated_variants["MAF_BIN"] = pd.Categorical(
        uncorrelated_variants["MAF_BIN"], categories=maf_bins,
        ordered=True)

    n_permuted_statistics = 1*10 ** 5

    np.random.seed(42)

    genpareto_parameters = dict()
    pperm_distributions = dict()
    pperm_statistics = dict()
    # maf_bins = np.unique(assoc["MAF"].round(2))
    for maf_bin in maf_bins:
        # Get variants of interest
        corrective_variants = uncorrelated_variants.loc[
            uncorrelated_variants["MAF_BIN"] == maf_bin]["SNP"]

        print("maf: {}, variants: {}".format(
            maf_bin, len(corrective_variants)))

        # From 0 to the number of rows in the association dataframe,
        # randomly select 1000 integers, which we'll use to select
        # variants from the assoc table.
        # number_of_variants_to_choose_from = assoc_maf_bin.shape[0]

        # row_numbers = np.random.randint(number_of_variants_to_choose_from,
        #                                 size=n_permuted_statistics)

        # Select the corresponding variants
        permutation_selection = assoc.loc[
            assoc["SNP"].isin(corrective_variants), "Z_perm_abs"].values

        pperm_distributions[maf_bin] = (
           fit_distribution(permutation_selection, anderson_darling_table))
        pperm_statistics[maf_bin] = permutation_selection
        # print(pperm_distributions[maf_bin][1])
        genpareto_parameters[maf_bin] = pperm_distributions[maf_bin][0].kwds

        plot_genpareto(
           standard_scores=pperm_statistics[maf_bin],
           genpareto_fit=pperm_distributions[maf_bin][0],
           n_exceedances=pperm_distributions[maf_bin][1],
           label=maf_bin)

    permutation_dataframe = pd.DataFrame({
        'MAF_BIN': np.repeat(list(pperm_statistics.keys()),
                            [len(v) for k, v in pperm_statistics.items()]),
        'Z_perm_abs': list(chain.from_iterable(pperm_statistics.values())),
    })

    sns.scatterplot(x="MAF_BIN", y="Z_perm_abs",
                    data=permutation_dataframe, linewidth=0)
    plt.show()

    plot_pdf_over_maf(pperm_distributions, pperm_statistics, maf_bins)

    accurate_permuted_pvalues = get_permutation_pvalues(
        empirical_dataframe=assoc,
        permutation_dataframe=permutation_dataframe,
        permutation_distributions=pperm_distributions,
        maf="MAF_BIN",
        test="Z_emp_abs",
        perm="Z_perm_abs"
    )

    ax = sns.countplot(accurate_permuted_pvalues["MAF_BIN"])
    ax.tick_params(axis='x', rotation=90)
    plt.show()

    ax = sns.countplot(permutation_dataframe["MAF_BIN"])
    ax.tick_params(axis='x', rotation=90)
    plt.show()

    accurate_permuted_pvalues["P_accurate_log10"] = -np.log10(
        accurate_permuted_pvalues["P_accurate"])
    accurate_permuted_pvalues["P_ecdf_psdoc_log10"] = -np.log10(
        accurate_permuted_pvalues["P_ecdf_pseudocount"])
    accurate_permuted_pvalues["P_gpd_log10"] = -np.log10(
        accurate_permuted_pvalues["P_gpd"])
    accurate_permuted_pvalues["P_emp_log10"] = -np.log10(
        accurate_permuted_pvalues["Pemp"])
    accurate_permuted_pvalues["MAF_BIN_I"] = accurate_permuted_pvalues["MAF_BIN"].cat.codes

    # In theory, the smaller the MAF, the further the tail should reach.

    grid = sns.FacetGrid(accurate_permuted_pvalues, col="MAF_BIN_I", col_wrap=5)

    grid.map(sns.scatterplot, "P_emp_log10", "P_accurate_log10",
             marker=".", alpha=0.5, linewidth=0)
    grid.fig.tight_layout(w_pad=1)

    grid.set(
        xlabel="empirical p-values (-log10 transformed)",
        ylabel="accurate perm p-values (-log10 transformed)")
    plt.show()

    comb = sns.scatterplot(
        data=accurate_permuted_pvalues,
        x="P_emp_log10", y="P_accurate_log10",
        hue="MAF_BIN_I", palette="viridis",
        marker=".", alpha=0.5, linewidth=0)
    color_mapping = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=0, vmax=19))
    color_mapping.set_array(accurate_permuted_pvalues["MAF_BIN_I"])
    comb.set(
        xlabel="empirical p-values (-log10 transformed)",
        ylabel="accurate perm p-values (-log10 transformed)")

    comb.get_legend().remove()
    comb.figure.colorbar(color_mapping)

    plt.show()


    grid = sns.FacetGrid(accurate_permuted_pvalues, col="MAF_BIN", col_wrap=5)

    grid.map(sns.scatterplot, "P_ecdf_psdoc_log10", "P_accurate_log10",
             marker=".", alpha=0.5, linewidth=0)
    grid.fig.tight_layout(w_pad=1)

    grid.set(
        xlabel="ecdf p-values (with psdoc; -log10 transformed)",
        ylabel="accurate perm p-values (-log10 transformed)")
    plt.show()


    comparison_plot = sns.scatterplot(
        x=-np.log10(accurate_permuted_pvalues["P_perm_accurate"]),
        y=-np.log10(accurate_permuted_pvalues["Pemp"]),
        hue=(accurate_permuted_pvalues["M"] >= 10),
        alpha=0.5, marker="+")
    comparison_plot.set(
        xlabel="accurate perm p-values (-log10 transformed)",
        ylabel="ECDF p-values (-log10 transformed)")
    plt.legend(loc='lower right')
    plt.show()

    # Fit a generalized pareto distribution to the permuted results.

    return 0


if __name__ == "__main__":
    sys.exit(main())
