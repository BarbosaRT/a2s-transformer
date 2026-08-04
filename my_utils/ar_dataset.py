import os
import librosa
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

from my_utils.ctc_dataset import CTCDataset, load_dataset, SPLITS, PIANO_HF_DATASET
from my_utils.data_preprocessing import (
    ar_batch_preparation,
    pad_batch_audios,
    preprocess_log_stft_audio,
    get_spectrogram_number_of_frames,
)

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

MUSICFM_SR = 24000
ENC_FRAMES_PER_SAMPLE = 960  # hop_length=240 * subsampling=4


def ar_val_batch_preparation_musicfm(batch):
    x, y = zip(*batch)
    xl = [xi.shape[-1] for xi in x]  # raw sample count; the model computes the frame count
    x = pad_batch_audios(x, dtype=torch.float32)
    xl = torch.tensor(xl, dtype=torch.int64)
    return x, xl, list(y)


def ar_val_batch_preparation_spectrogram(batch):
    x, y = zip(*batch)
    xl = [get_spectrogram_number_of_frames(xi.shape[-1]) for xi in x]
    x = pad_batch_audios(x, dtype=torch.float32)
    xl = torch.tensor(xl, dtype=torch.int64)
    return x, xl, list(y)


class ARDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        use_voice_change_token: bool = False,
        batch_size: int = 16,
        num_workers: int = None,
        audio_mode: str = "musicfm",
        tokenization: str = "kern",
    ):
        super(ARDataModule, self).__init__()
        self.ds_name = ds_name
        self.use_voice_change_token = use_voice_change_token
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 4)
        self.audio_mode = audio_mode
        self.tokenization = tokenization

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _val_collate_fn(self):
        if self.audio_mode == "spectrogram":
            return ar_val_batch_preparation_spectrogram
        return ar_val_batch_preparation_musicfm

    def setup(self, stage: str):
        if stage == "fit":
            if not self.train_ds:
                self.train_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="train",
                    use_voice_change_token=self.use_voice_change_token,
                    audio_mode=self.audio_mode,
                    tokenization=self.tokenization,
                )
            if not self.val_ds:
                self.val_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="val",
                    use_voice_change_token=self.use_voice_change_token,
                    audio_mode=self.audio_mode,
                    tokenization=self.tokenization,
                )

        if stage == "test" or stage == "predict":
            if not self.test_ds:
                self.test_ds = ARDataset(
                    ds_name=self.ds_name,
                    partition_type="test",
                    use_voice_change_token=self.use_voice_change_token,
                    audio_mode=self.audio_mode,
                    tokenization=self.tokenization,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ar_batch_preparation,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._val_collate_fn(),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._val_collate_fn(),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        print("Using test_dataloader for predictions.")
        return self.test_dataloader()

    def get_w2i_and_i2w(self):
        try:
            return self.train_ds.w2i, self.train_ds.i2w
        except AttributeError:
            return self.test_ds.w2i, self.test_ds.i2w

    def get_max_seq_len(self):
        try:
            return self.train_ds.max_seq_len
        except AttributeError:
            return self.test_ds.max_seq_len

    def get_max_audio_len(self):
        try:
            return self.train_ds.max_audio_len
        except AttributeError:
            return self.test_ds.max_audio_len


####################################################################################################


class ARDataset(CTCDataset):
    def __init__(
        self,
        ds_name: str,
        partition_type: str,
        use_voice_change_token: bool = False,
        audio_mode: str = "musicfm",
        tokenization: str = "kern",
    ):
        self.ds_name = ds_name.lower()
        self.partition_type = partition_type
        self.use_voice_change_token = use_voice_change_token
        self.audio_mode = audio_mode
        self.tokenization = tokenization
        self.init(vocab_name="ar_w2i")
        self.max_seq_len += 1  # Add 1 for EOS_TOKEN

    def __getitem__(self, idx):
        audio = self.ds[idx]["audio"]
        raw = audio["array"]
        sr = audio["sampling_rate"]
        if self.audio_mode == "spectrogram":
            x = preprocess_log_stft_audio(raw_audio=raw, sr=sr, dtype=torch.float32)
        else:
            raw = librosa.resample(raw, orig_sr=sr, target_sr=MUSICFM_SR)
            x = torch.from_numpy(raw).unsqueeze(0).float()
        y = self.preprocess_transcript(self.ds[idx])
        if self.partition_type == "train":
            return x, self.get_number_of_frames(x), y
        return x, y

    def preprocess_transcript(self, sample):
        tokens = super().preprocess_transcript(sample)
        tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
        return torch.tensor([self.w2i[w] for w in tokens], dtype=torch.int64)

    def make_vocabulary(self):
        hf_name = PIANO_HF_DATASET if self.is_piano else f"PRAIG/{self.ds_name}-quartets"
        full_ds = load_dataset(hf_name)

        vocab = []
        for split in SPLITS:
            for sample in full_ds[split]:
                transcript = super().preprocess_transcript(sample)
                vocab.extend(transcript)
        vocab = [SOS_TOKEN, EOS_TOKEN] + vocab
        vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    def get_number_of_frames(self, audio):
        if self.audio_mode == "spectrogram":
            return get_spectrogram_number_of_frames(audio.shape[-1])
        return audio.shape[1]
