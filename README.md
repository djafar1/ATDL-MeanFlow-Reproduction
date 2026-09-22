# ATDL Group Assignment: MeanFlow Reproduction

Reproducibility project for the course Advanced Topics in Deep Learning at the University of Copenhagen.

## Group members

- Said Djafar Said
- Simon Winther Albertsen
- Hjalte Svaneborg Bjørnstrup

## Paper

**Mean Flows for One-step Generative Modeling**  
Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, Kaiming He

Paper: https://arxiv.org/abs/2505.13447  
Official code: https://github.com/Gsunshine/meanflow

## Repository structure

The original MeanFlow implementation is kept under `src/meanflow`.

Our own experiments, configurations, scripts and results are kept separately:

- `src/meanflow/` original MeanFlow implementation
- `experiments/` our reproduction experiments
- `configs/` experiment configurations
- `scripts/` helper and evaluation scripts
- `results/` generated results, figures and tables
- `environment/` environment and dependency setup

This separation makes it easier to keep track of what comes from the original implementation and what was added or changed as part of our reproducibility study.
