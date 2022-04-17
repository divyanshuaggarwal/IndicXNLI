from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import torch


def get_trainer(fast_dev=True):
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode="min",
    )
    return Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16,
        # tpu_cores = 8, #Uncoment this if using TPU.
        accelerator="auto",
        val_check_interval=0.5,
        callbacks=[early_stop_callback],
        fast_dev_run=fast_dev,
    )
