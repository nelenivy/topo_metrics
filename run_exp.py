from run_models import ModelKeeper
from run_metrics import evaluate_one_emb
from time import time
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import gc
import os
import torch

def clear_checkpoints_dir(checkpoints_path):
    if not os.path.exists(checkpoints_path):
        return
    for f in os.listdir(checkpoints_path):
        fp = os.path.join(checkpoints_path, f)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                import shutil
                shutil.rmtree(fp)
        except Exception as e:
            print(f"{fp} is not deleted ({e})")


def create_params_grid(fixed_params, variable_params):
    # Создание списка всех гиперпараметров, которые нужно перебрать
    all_hyperparameter_grids = []
    for variable_param_name, variable_param_values in variable_params.items():
        for value in variable_param_values:
            hyperparameter_grid = {**fixed_params, variable_param_name: value}
            all_hyperparameter_grids.append((variable_param_name, hyperparameter_grid))

    return all_hyperparameter_grids

def run_grid_search(all_hyperparameter_grids, sample_fractions,
        train_data_in, valid_data_in, test_data_in, targets,
        checkpoints_path, logger,col_id="customer_id", 
        target_col='gender', out_prefix=None,verbose=0, n_samples=10):
    cur_time = time()
    all_embs = []    

    for param in tqdm(all_hyperparameter_grids):            
        logger.info(f'All params are frozen except {param[0]}')
        params = param[1]
        model_keeper = ModelKeeper()    
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Testing parameters: {params}")        
        model_keeper.create_datasets(train_data_in, valid_data_in, params, 
                            col_id=col_id)
        model_keeper.train_model(params, checkpoints_path=checkpoints_path)
       
        embs = model_keeper.calc_embs_from_trained(test_data_in)
        
        clear_checkpoints_dir(checkpoints_path=checkpoints_path)
        
        all_embs += embs

    eval_many_embs(all_embs, targets, 
        col_id=col_id, target_col=target_col, out_prefix=out_prefix, 
        sample_fractions=sample_fractions, verbose=verbose, n_samples=n_samples)

def eval_many_embs(embs_list, targets, col_id='customer_id', 
        target_col='gender', out_prefix=None, 
        sample_fractions=tuple((1/20,)), verbose=0, n_samples=10): 
    res_per_sample_frac = defaultdict(list) 

    for curr_emb in tqdm(embs_list): 
        res = evaluate_one_emb(curr_emb['emb'], targets, 
            sample_fractions=sample_fractions,
            col_id=col_id, target_col=target_col, 
            verbose=verbose, n_samples=n_samples)
    
        for i, metrics in enumerate(res):
            metrics_flattened = {k: round(v, 4) for k, v in metrics.items()}
            #times_flattened = {f"time_{k}": round(v, 4) for k, v in times.items()}
            # Сохранение результатов
            res_dict = {
                **curr_emb['info'],
                **metrics_flattened
                #**times_flattened,
            }

            res_per_sample_frac[metrics['sample_fraction']].append(res_dict)

    # Сохранение в CSV
    for sample_frac, new_result in res_per_sample_frac.items():
        print(sample_frac)
        print(new_result)
        new_result = pd.DataFrame(new_result)

        # if not os.path.exists(output_csv):  
        #     pd.DataFrame(columns=columns).to_csv(output_csv, mode="w", index=False, header=True)
        output_csv = f"{out_prefix}_{sample_frac:.3f}".rstrip('0').rstrip('.') + ".csv"

        new_result.to_csv(output_csv, mode="w")#, header=False, index=False)

