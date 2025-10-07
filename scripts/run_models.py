import glob

from functools import partial

import pandas as pd
import torch
import pytorch_lightning as pl

from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames import PtlsDataModule
from ptls.frames.coles import CoLESModule
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames.coles.losses import (
    BarlowTwinsLoss,
    ContrastiveLoss,
    VicregLoss,
    SoftmaxLoss
)
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.datasets import MemoryMapDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from ptls.frames.inference_module import InferenceModule
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles.sampling_strategies import HardNegativePairSelector


class CustomLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.early_stopping_epoch = None
        self.topk_list = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)
        val_loss = trainer.callback_metrics.get("val_loss", None)
        curr_recall_topk = trainer.callback_metrics.get(
            "valid/recall_top_k", None)
        if curr_recall_topk:
            curr_recall_topk = curr_recall_topk.cpu().numpy()
        self.topk_list.append(curr_recall_topk)
        if train_loss is not None and val_loss is not None:
            print(
                f"Epoch {trainer.current_epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Если валидационный лосс увеличивается - фиксируем эпоху остановки
        if trainer.early_stopping_callback is not None and trainer.early_stopping_callback.wait_count == 0:
            self.early_stopping_epoch = trainer.current_epoch


class DataPrepare:
    def __init__(self, **kwargs):
        pass


class ModelKeeper:
    def __init__(self, **kwargs):
        self.losses = {
            "BarlowTwinsLoss": BarlowTwinsLoss(
                lambd=0.
            ),
            "ContrastiveLoss": ContrastiveLoss(
                margin=0.5,
                sampling_strategy=HardNegativePairSelector(neg_count=5)
            ),
            "VicregLoss": VicregLoss(
                sim_coeff=25,
                std_coeff=25,
                cov_coeff=1
            ),
            "SoftmaxLoss": SoftmaxLoss(
                temperature=0.07
            ),
        }

    def create_datasets(self, train_data_in, valid_data_in, params, col_id="customer_id"):
        self.col_id = col_id

        splitter = SampleSlices(
            split_count=(
                2*(params["split_count"] // 2)
                if params["loss"] in ["BarlowTwinsLoss", "VicregLoss"]
                else params["split_count"]
            ),
            cnt_min=params["cnt_min"],
            cnt_max=params["cnt_max"],
        )

        train_data = ColesDataset(
            MemoryMapDataset(
                data=train_data_in,
                i_filters=[SeqLenFilter(min_seq_len=25)],
            ),
            splitter=splitter
        )

        valid_data = ColesDataset(
            MemoryMapDataset(
                data=valid_data_in,
                i_filters=[SeqLenFilter(min_seq_len=25)],
            ),
            splitter=splitter
        )

        self.train_loader = PtlsDataModule(
            train_data=train_data,
            train_batch_size=params["batch_size"],
            train_num_workers=0,
            valid_data=valid_data,
        )

    def curr_checkpoint_name(self):
        return f"model_{self.params['batch_size']}_{self.params['learning_rate']}" \
            f"_{self.params['split_count']}_{self.params['cnt_min']}_{self.params['cnt_max']}" \
            f"_{self.params['hidden_size']}_{self.params['embedding_dim']}" \
            f"_{self.params['category_embedding_dim']}_{self.params['hidden_size']}" \
            f"_{self.params['loss']}_{self.params['rnn_encoder_type']}"

    def train_model(self, params, checkpoints_path, recalculate=False, devices=0):
        self.params = params
        self.checkpoints_path = checkpoints_path

        trx_encoder_params = dict(
            embeddings_noise=0.003,
            linear_projection_size=params['embedding_dim'],
            embeddings={
                "mcc_code": {
                    "in": params['mcc_code_in'],
                    "out": params['category_embedding_dim']
                },
                "term_id": {
                    "in": params['term_id_in'],
                    "out": params['category_embedding_dim']
                },
                "tr_type": {
                    "in": params['tr_type_in'],
                    "out": params['category_embedding_dim']
                },
            },
            numeric_values={"amount": "identity"},
        )

        trx_encoder = TrxEncoder(**trx_encoder_params)

        self.seq_encoder = RnnSeqEncoder(
            trx_encoder=trx_encoder,
            input_size=params['embedding_dim'],

            # Используем только текущее значение hidden_size
            hidden_size=self.params["hidden_size"],
            type=params['rnn_encoder_type']
        )

        self.model = CoLESModule(
            seq_encoder=self.seq_encoder,
            optimizer_partial=partial(
                torch.optim.Adam,
                lr=self.params["learning_rate"]
            ),
            lr_scheduler_partial=partial(
                torch.optim.lr_scheduler.StepLR,
                step_size=10,
                gamma=0.5
            ),
            loss=self.losses.get(
                params["loss"],
                None
            )
        )

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoints_path,
            filename=f"{self.curr_checkpoint_name()}{{epoch:02d}}",
            save_top_k=-1,
            every_n_epochs=1,
        )

        custom_logger = CustomLogger()

        # Обучение модели
        self.pl_trainer = pl.Trainer(
            callbacks=[
                checkpoint_callback,
                custom_logger
            ],
            default_root_dir=self.checkpoints_path,
            check_val_every_n_epoch=1,
            max_epochs=params['num_epochs'],
            accelerator="gpu",
            devices=devices,
            enable_progress_bar=True
        )

        print(self.curr_checkpoint_name())

        checkpoint_files = glob.glob(
            f"{self.checkpoints_path}/{self.curr_checkpoint_name()}*.ckpt"
        )

        print(checkpoint_files)

        self.topk_list = []

        if (len(checkpoint_files) == 0) or recalculate:
            self.model.train()
            self.pl_trainer.fit(self.model, self.train_loader)

            self.early_stop_epoch = custom_logger.early_stopping_epoch
            self.topk_list = custom_logger.topk_list

            if self.early_stop_epoch is None:
                self.early_stop_epoch = params['num_epochs']

        else:
            self.early_stop_epoch = params['num_epochs']

    def calc_embs_from_trained(self, test_data):
        inference_dataset = MemoryMapDataset(
            data=test_data,
        )

        checkpoint_files = glob.glob(
            f"{self.checkpoints_path}/{self.curr_checkpoint_name()}*.ckpt"
        )
        checkpoint_files.sort()

        res = []

        for i, checkpoint in enumerate(checkpoint_files):
            self.model = CoLESModule.load_from_checkpoint(
                checkpoint, seq_encoder=self.seq_encoder)

            self.model.eval()

            inference_module = InferenceModule(
                model=self.model,
                pandas_output=True,
                drop_seq_features=True,
                model_out_name='emb'
            )

            inference_dl = torch.utils.data.DataLoader(
                dataset=inference_dataset,
                collate_fn=collate_feature_dict,
                shuffle=False,
                batch_size=64,
                num_workers=16,
            )

            inference_module.model.is_reduce_sequence = True

            inf_test_embeddings = pd.concat(
                self.pl_trainer.predict(inference_module, inference_dl),
                axis=0,
            )

            print(len(inf_test_embeddings))

            res.append({"emb": inf_test_embeddings, "info": {
                **self.params,
                "checkpoint": checkpoint,
                "epoch_num": int(i),
                "early_stop_epoch": int(self.early_stop_epoch),
                "recall_topk": self.topk_list[i] if self.topk_list else 0.0}
            })

        return res
