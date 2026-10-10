# ISBI 2021 Paper Submission Templates
Word and Latex template documents for [ISBI 2021](https://biomedicalimaging.org/2021/) submissions can be found here.

# Assignment report

Group 11 submits under the Replication Track. The assignment repository is
[ATDL-MeanFlow-Reproduction](https://github.com/djafar1/ATDL-MeanFlow-Reproduction),
confirmed by Simon and configured as this repository's `origin`. The separate
[authors' MeanFlow repository](https://github.com/Gsunshine/meanflow) supplies the
JAX implementation for Simon's experiment.

The experiment assignments, confirmed by Simon, are:

- Simon Winther Albertsen: Table 1(e), adaptive loss metrics; the ImageNet-256/JAX
  experiment is integrated with ten verified runs and two seeds per exponent.
- Hjalte Bjoernstrup: Table 1(b), Jacobian-vector-product computation.
- Said Djafar Said: Table 1(a), the ratio of sampling r != t.

Hjalte's and Djafar's experiment sections, execution details and numerical results
are not present in this checkout. The report credits their confirmed experiment
contributions and leaves source TODO comments for integrating their results and
updating the shared abstract, discussion and conclusion. Their datasets and code
must be documented from their protocols, rather than inferred from the existing
draft's unsupported CIFAR-10/PyTorch description or the paper's ImageNet targets.

The abstract reports only the verified loss-metric findings. Specific contribution
statements, the group number, track and clickable assignment repository link
are included on the front page. The existing author names and template settings
are preserved. The assignment instructions are in `../Group_Assignment.md`.

# Usage
Please refer to the [IEEE Author Center](https://ieeeauthorcenter.ieee.org/publish-with-ieee/author-education-resources/guidelines-and-policies/policy-posting-your-article/) for usage guidelines on the templates. Specific intructions for ISBI 2021 submission can be found on the [conference website](https://biomedicalimaging.org/2021/full-paper-instructions/).

# Licence
This material is subject to the LaTeX Project Public License. See http://www.ctan.org/license/lppl1.3 for the details of that license.
