## Overview
This repository supports experimental investigation of unsupervised embedding metrics, emphasizing topological measures, applied to financial and recommender system datasets. It facilitates parameter grid search and results aggregation for metric evaluation in complex analytic setups.

## Main Entry Points

- **sample_frac_results_all.ipynb**  
  Notebook for aggregation and analysis of experimental results in the financial analytics context, enabling comprehensive metric comparison.

- **recsys_challenge/gru/run.sh**  
  Script to execute parameter grid search experiments for the recommender system challenge using GRU model settings.

- **financial_analytics/age_pred/age_pred_script_device_0.py and age_pred_script_device_1.py**  
  Python scripts for distributed grid search experiments on the age prediction dataset, designed for device-specific execution.

- **financial_analytics/gender/gender_script_device_0.py and gender_script_device_1.py**  
  Similar distributed experiment scripts targeting the gender prediction dataset.

## Results
Processed experiment outputs in CSV format are organized within the **csv_results/** directory, structured to facilitate downstream analysis.
