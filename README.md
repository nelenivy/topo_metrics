# Topological Metric for Unsupervised Embedding Quality Evaluation
## Overview
This repository contains the experiments for the article "Topological Metric for Unsupervised Embedding Quality Evaluation." It enables an experimental investigation of unsupervised embedding metrics, with a focus on topological metrics applied to financial and recommender system datasets. The repository supports hyperparameter optimization, optimal epoch selection, and results aggregation for metrics evaluation across various configurations.

## Main Entry Points
### Behavioral modeling
- **recsys_challenge/gru/run.sh**  
  Script for executing hyperparameter optimization experiments on the RecSys Challenge 2025 using GRU model configurations.

### Financial analytics
- **financial_analytics/age_pred/age_pred_script_device_0\1.py** and **financial_analytics/gender/gender_script_device_0\1.py**  
  Python scripts for hyperparameter optimization and optimal epoch selection experiments on the age and gender prediction datasets within the financial analytics domain, designed for execution on specific devices.

### Collaborative filtering
- **collaborative_filtering/run_als.py** and **collaborative_filtering/run_bpr.py** 
  Python scripts for hyperparameter optimization on the Movielens-20m dataset using collaborative filtering algorithms (ALS and BPR).
  
## Results
 - Processed experiment outputs in CSV format are organized within the **csv_results/** directory, structured to facilitate downstream analysis.

- **sample_frac_results_all.ipynb**  
  Notebook for aggregating and analyzing experimental results in the financial analytics domain, enabling comparison of metrics for hyperparameter optimization and optimal epoch selection experiments based on correlation values and the best configuration quality scores.
