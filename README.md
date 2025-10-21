# Topological Metric for Unsupervised Embedding Quality Evaluation
## Overview
This repository contains the experiments for the article "Topological Metric for Unsupervised Embedding Quality Evaluation." It enables an experimental investigation of unsupervised embedding metrics, with a focus on topological metrics applied to financial and recommender system datasets. The repository supports parameter hyperparameter optimization, optimal epoch selection, and results aggregation for metrics evaluation across various configurations.

## Main Entry Points
### Behavioral modeling
- **recsys_challenge/gru/run.sh**  
  Script for executing hyperparameter optimization experiments on the recommender system challenge using GRU model configurations.

### Financial analytics
- **financial_analytics/age_pred/age_pred_script_device_0.py and age_pred_script_device_1.py**
- **financial_analytics/gender/gender_script_device_0.py and gender_script_device_1.py**
  
 Python scripts for hyperparameter optimization and optimal epoch selection experiments on the age and gender prediction datasets, designed for execution on specific devices.

## Results
 - Processed experiment outputs in CSV format are organized within the **csv_results/** directory, structured to facilitate downstream analysis.

- **sample_frac_results_all.ipynb**  
  Notebook for aggregation and analysis of experimental results in the financial analytics context, enabling metric comparison for hyperparameter optimization and optimal epoch selection experiments.
