import os

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from intrinsic_dim import compute_intrinsic_dim_global


@hydra.main(version_base=None, config_path="config", config_name="global")
def main(config: DictConfig):

    print(OmegaConf.to_yaml(config))

    embeddings_path = config.embeddings_path

    print(f"Looking for embeddings in: {embeddings_path}")
    if not os.path.exists(embeddings_path):
        print(f"Error: Path {embeddings_path} does not exist.")
        return

    embedding_dirs = [f for f in os.listdir(embeddings_path)
                      if os.path.isdir(os.path.join(embeddings_path, f))]
    print(f"Found {len(embedding_dirs)} directories.")

    results = {}

    for embedding_dir in tqdm(embedding_dirs):

        embedding_file = os.path.join(embeddings_path, embedding_dir, 'embeddings.npy')
        if not os.path.exists(embedding_file):
            print(f"Warning: embeddings.npy not found in {embedding_dir}")
            continue

        try:
            embeddings = np.load(embedding_file)
            if config.sample_size is not None:
                permutation_indices = np.random.permutation(embeddings.shape[0])
                embeddings = embeddings[permutation_indices[:config.sample_size]]

            results[embedding_dir] = compute_intrinsic_dim_global(
                embeddings, estimator_names=list(config.estimator_names))

        except Exception as e:
            print(f"Error processing {embedding_dir}: {e}")

    results_df = pd.DataFrame(results).T
    print(results_df)

    results_df.to_csv(config.output_file)
    print(f"Results saved to {config.output_file} in {os.getcwd()}")


if __name__ == "__main__":

    main()
