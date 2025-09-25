import pandas as pd
import numpy as np
from ptls.frames.inference_module import InferenceModuleMultimodal
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import gc
from time import time
import glob
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from functools import partial
from sklearn.model_selection import train_test_split

from ptls.nn import TrxEncoder
from ptls.nn.seq_encoder.rnn_encoder import RnnEncoder
from ptls.frames import PtlsDataModule
from ptls.frames.coles import CoLESModule
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames.coles.multimodal_dataset import MultiModalDataset
from ptls.frames.coles.multimodal_dataset import MultiModalIterableDataset
from ptls.frames.coles.multimodal_dataset import MultiModalSortTimeSeqEncoderContainer
from ptls.frames.coles.multimodal_inference_dataset import MultiModalInferenceIterableDataset
#from ptls.frames.coles.multimodal_inference_dataset import #MultiModalIterableDataset
from ptls.frames.inference_module import InferenceModuleMultimodal
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.data_load import IterableProcessingDataset
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.datasets import MemoryMapDataset
from ptls.preprocessing import PandasDataPreprocessor

class CustomLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.early_stopping_epoch = None
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)
        val_loss = trainer.callback_metrics.get("val_loss", None)
        
        if train_loss is not None and val_loss is not None:
            print(f"Epoch {trainer.current_epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Если валидационный лосс увеличивается - фиксируем эпоху остановки
        if trainer.early_stopping_callback is not None and trainer.early_stopping_callback.wait_count == 0:
            self.early_stopping_epoch = trainer.current_epoch

class DataPrepare:
    def __init__(self, **kwargs):
        pass

class ModelKeeper:
    def __init__(self, **kwargs):
        pass

    def create_datasets(self, train_data_in, valid_data_in, params, 
                            source_features, col_id="customer_id"):
        self.col_id = col_id
        self.source_features = source_features
        splitter = SampleSlices(
            split_count=params["split_count"],
            cnt_min=params["cnt_min"],
            cnt_max=params["cnt_max"],
        )

        train_data = MultiModalIterableDataset(
            data=train_data_in,
            splitter=splitter,
            source_features=source_features,
            col_id=col_id,
            col_time="event_time",
            source_names=("sourceA", "sourceB"),
        )

        valid_data = MultiModalIterableDataset(
            data=valid_data_in,
            splitter=splitter,
            source_features=source_features,
            col_id=col_id,
            col_time="event_time",
            source_names=("sourceA", "sourceB"),
        )

        self.train_loader = PtlsDataModule(
            train_data=train_data,
            train_batch_size=params["batch_size"],
            train_num_workers=0,
            valid_data=valid_data,
        )

    def train_model(self, params, mcc_code_in, term_id_in, tr_type_in, num_epochs, checkpoints_path):
        self.params = params        
        self.checkpoints_path = checkpoints_path

        sourceA_encoder_params = dict(
            embeddings_noise=0.003,
            linear_projection_size=64,
            embeddings={
                "mcc_code": {"in": mcc_code_in, "out": 32},
                "term_id": {"in": term_id_in, "out": 32},
            },
        )
        
        sourceB_encoder_params = dict(
            embeddings_noise=0.003,
            linear_projection_size=64,
            embeddings={
                "tr_type": {"in": tr_type_in, "out": 32},
            },
            numeric_values={"amount": "identity"},
        )
        
        sourceA_encoder = TrxEncoder(**sourceA_encoder_params)
        sourceB_encoder = TrxEncoder(**sourceB_encoder_params)
        
        self.seq_encoder = MultiModalSortTimeSeqEncoderContainer(
            trx_encoders={
                "sourceA": sourceA_encoder,
                "sourceB": sourceB_encoder,
            },
            input_size=64,
            hidden_size=self.params["hidden_size"],  # Используем только текущее значение hidden_size
            seq_encoder_cls=RnnEncoder,
            type="gru",
        )

        self.model = CoLESModule(
            seq_encoder=self.seq_encoder,
            optimizer_partial=partial(torch.optim.Adam, lr=self.params["learning_rate"]),
            lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.5),
        )


        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoints_path,
            filename=f"model_{self.params['batch_size']}_{self.params['learning_rate']}_{self.params['split_count']}_{self.params['cnt_min']}_{self.params['cnt_max']}_{self.params['hidden_size']}{{epoch:02d}}",
            save_top_k=-1,
            every_n_epochs=1,
        )

        custom_logger = CustomLogger()
        early_stopping_callback = EarlyStopping(
            monitor="valid/recall_top_k",#"val_loss"
            patience=5,
            mode="max", #"min",
            verbose=True
        )
        # Обучение модели
        self.pl_trainer = pl.Trainer(
            callbacks=[checkpoint_callback, early_stopping_callback, custom_logger],
            default_root_dir=self.checkpoints_path,
            check_val_every_n_epoch=1,
            max_epochs=num_epochs,
            accelerator="gpu",
            devices=1,
            enable_progress_bar=True,
            precision='bf16-mixed'
        )
        self.model.train()
        self.pl_trainer.fit(self.model, self.train_loader)

        self.early_stop_epoch = custom_logger.early_stopping_epoch

        if self.early_stop_epoch is None:
            self.early_stop_epoch = num_epochs

    def calc_embs_from_trained(self, test_data, model_out_name="emb"):
        inf_test_data = MultiModalInferenceIterableDataset(
            data = test_data,
            source_features = self.source_features,
            col_id = self.col_id,
            col_time = "event_time",
            source_names = ("sourceA", "sourceB")
        )

        # Обработка чекпоинтов
        checkpoint_files = glob.glob(f"{self.checkpoints_path}/model_{self.params['batch_size']}_{self.params['learning_rate']}_{self.params['split_count']}_{self.params['cnt_min']}_{self.params['cnt_max']}_{self.params['hidden_size']}*.ckpt")
        checkpoint_files.sort()
        #logger.info(f"Elapsed time: {time() - cur_time:.2f} seconds")

        #logger.info(f'Early stop is {self.early_stop_epoch}')
        res = []
        #print(checkpoint_files)
        for i, checkpoint in enumerate(checkpoint_files):
            #logger.info(f"Processing checkpoint number {i}")
            self.model = CoLESModule.load_from_checkpoint(checkpoint, seq_encoder=self.seq_encoder)

            # Вычисление метрик и времени
            self.model.eval()
            inference_module = InferenceModuleMultimodal(
                model=self.model,
                pandas_output=True,
                drop_seq_features=True,
                model_out_name=model_out_name,
                col_id=self.col_id,
            )
            inf_test_loader = DataLoader(
                dataset = inf_test_data,
                collate_fn = partial(inf_test_data.collate_fn, col_id=self.col_id),
                shuffle = False,
                num_workers = 0,
                batch_size = 8
                )
            inference_module.model.is_reduce_sequence = True

            # Получение эмбеддингов
            inf_test_embeddings = pd.concat(
                self.pl_trainer.predict(inference_module, inf_test_loader),
                axis=0,
            )
            print(inf_test_embeddings.shape)
            print(np.unique(inf_test_embeddings[self.col_id].unique().shape))
            res.append({"emb": inf_test_embeddings, "info":{
                    **self.params,
                    "checkpoint": checkpoint,
                    "epoch_num": int(i),
                    "early_stop_epoch": int(self.early_stop_epoch)}
                })

        torch.cuda.empty_cache()
        gc.collect()
        return res




    