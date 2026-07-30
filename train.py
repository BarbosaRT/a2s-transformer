import gc
import os
import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger

from networks.crnn.model import CTCTrainedCRNN
from networks.transformer.model import A2STransformer
from networks.transformer2.model import A2STransformer as A2STransformer2
from my_utils.ctc_dataset import CTCDataModule
from my_utils.ar_dataset import ARDataModule
from my_utils.seed import seed_everything

torch.set_float32_matmul_precision("high")
seed_everything(42, benchmark=False)


def train(
    ds_name, use_checkpoint: bool = False,
    model_type: str = "crnn",
    attn_window: int = -1,
    use_voice_change_token: bool = False,
    epochs: int = 1000,
    patience: int = 20,
    batch_size: int = 16,
    check_val_every_n_epoch: int = 5,
    is_flash: bool = False,
    pretrained_path: str = None,
    freeze_encoder: bool = False,
    strategy: str = "ddp_find_unused_parameters_true",
):
    gc.collect()
    torch.cuda.empty_cache()

    # Experiment info
    print("TRAIN EXPERIMENT")
    print(f"\tDataset: {ds_name}")
    print(f"\tModel type: {model_type}")
    print(f"\tAttention window: {attn_window} (Used if model type is transformer/transformer2)")
    print(f"\tUse voice change token: {use_voice_change_token}")
    print(f"\tIs flash: {is_flash}")
    print(f"\tPretrained path: {pretrained_path}")
    print(f"\tFreeze encoder: {freeze_encoder}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tBatch size: {batch_size}")
    print(f"\tCheck Val Every N epoch: {check_val_every_n_epoch}")

    if model_type == "crnn":
        # Data module
        datamodule = CTCDataModule(
            ds_name=ds_name,
            use_voice_change_token=use_voice_change_token,
            batch_size=batch_size,
        )
        datamodule.setup(stage="fit")
        w2i, i2w = datamodule.get_w2i_and_i2w()

        # Model
        model = CTCTrainedCRNN(
            w2i=w2i,
            i2w=i2w,
            max_audio_len=datamodule.get_max_audio_len(),
            frame_multiplier_factor=datamodule.get_frame_multiplier_factor(),
        )
        # Override the datamodule width reduction factors with that of the model
        datamodule.width_reduction = model.width_reduction

    elif model_type == "transformer":
        # Data module
        datamodule = ARDataModule(
            ds_name=ds_name,
            use_voice_change_token=use_voice_change_token,
            batch_size=batch_size,
        )
        datamodule.setup(stage="fit")
        w2i, i2w = datamodule.get_w2i_and_i2w()

        # Model
        model = A2STransformer(
            max_seq_len=datamodule.get_max_seq_len(),
            max_audio_len=datamodule.get_max_audio_len(),
            w2i=w2i,
            i2w=i2w,
            attn_window=attn_window,
            teacher_forcing_prob=0.2,
        )

    elif model_type == "transformer2":
        # Data module
        datamodule = ARDataModule(
            ds_name=ds_name,
            use_voice_change_token=use_voice_change_token,
            batch_size=batch_size,
        )
        datamodule.setup(stage="fit")
        w2i, i2w = datamodule.get_w2i_and_i2w()

        # Model with MusicFM encoder
        model = A2STransformer2(
            max_seq_len=datamodule.get_max_seq_len(),
            max_audio_len=datamodule.get_max_audio_len(),
            w2i=w2i,
            i2w=i2w,
            attn_window=attn_window,
            teacher_forcing_prob=0.2,
            is_flash=is_flash,
            pretrained_path=pretrained_path,
            freeze_encoder=freeze_encoder,
        )

        # Pre-compute encoder outputs once if frozen
        if freeze_encoder:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.encoder.eval()
            train_loader = datamodule.train_dataloader()
            cached_data = []
            with torch.no_grad():
                for batch in train_loader:
                    x, xl, y_in, y_out = batch
                    x = x.to(device)
                    xl = xl.to(device)
                    enc_out = model.encoder(x)
                    xl_new = torch.ceil(xl.float() / 4).long()
                    cached_data.append((enc_out.cpu(), xl_new.cpu(), y_in, y_out))
            print(f"Cached {len(cached_data)} training batches of encoder outputs")

            class CachedDataset(torch.utils.data.Dataset):
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]

            def collate_cached(batch):
                return batch[0]

            cached_loader = torch.utils.data.DataLoader(
                CachedDataset(cached_data),
                batch_size=1,
                shuffle=True,
                collate_fn=collate_cached,
            )
            datamodule.train_dataloader = lambda: cached_loader

    else:
        print(f"Model type {model_type} not implemented")
        raise NotImplementedError

    # Train, validate and test
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{model_type}" if not use_voice_change_token else f"weights/{model_type}-VCT",
            filename=ds_name,
            monitor="val_sym-er",
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=5,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor="val_sym-er",
            min_delta=0.01,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=1000.00,
            check_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    trainer = Trainer(
        log_every_n_steps=100,
        logger=WandbLogger(
            project="A2S-Poly-ICASSP",
            group=f"{model_type}" if not use_voice_change_token else f"{model_type}-VCT",
            name=f"Train-{ds_name}_Test-{ds_name}",
            log_model=False,
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
        deterministic=False,  # If True, raises error saying that CTC loss does not have this behaviour
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
        strategy=strategy,
    )
    ckpt_path = f"weights/{model_type}/{ds_name}.ckpt"
    resume_ckpt = ckpt_path if os.path.exists(ckpt_path) else None
    if resume_ckpt and use_checkpoint:
        print(f"Resuming from checkpoint: {resume_ckpt}")
        trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    else:
        print("No checkpoint found, starting fresh.")    
        trainer.fit(model, datamodule=datamodule)
    
    if model_type == "crnn":
        model = CTCTrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)
    elif model_type == "transformer2":
        model = A2STransformer2.load_from_checkpoint(callbacks[0].best_model_path)
    else:
        model = A2STransformer.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)
