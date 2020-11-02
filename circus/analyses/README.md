# Analyses

This subpackage contains usual analyses made on the results of the sorting procedure.

To run one of these analyses, we recommend to invoke a IPython interpreter to have access to the variables defined during the execution of this script.

For example, to display a given template, execute the following commands in a terminal:
```
$ ipython
In [1]: %run -m circus.analyses.display_template <datafile> -t <template_id>
In [2]: plt.show()
```
where `<datafile>` and `<template_id` need to be replace by appropriate values.

You can show the help message associated to this analysis with:
```
In [1]: %run -m circus.analyses.display_template -h
```
This holds for other analysis.

Here is a list of the analyses present in this subpackage:
- `circus.analyses.display_template`
- `circus.analyses.display_isi`
- `circus.analyses.display_amplitudes`
- whitening
  - `circus.analyses.whitening.inspect_thresholds`
  - `circus.analyses.whitening.spatial_matrix`
  - `circus.analyses.whitening.collected_waveforms`
  - `circus.analyses.whitening.basis_waveforms`
- clustering
  - `circus.analyses.clustering.inspect_clusters_projections`
  - `circus.analyses.clustering.inspect_clusters_snippets`
- fitting
  - `circus.analyses.fitting.inspect_amplitude_pdfs`
  - `circus.analyses.fitting.inspect_fitted_snippets`
  - `circus.analyses.fitting.inspect_autocorrelograms`
  - `circus.analyses.fitting.inspect_rpvs`
