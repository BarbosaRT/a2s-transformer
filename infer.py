import math
import os
import tempfile

import fire
import librosa
import numpy as np
import torch

from my_utils.ar_dataset import EOS_TOKEN, SOS_TOKEN
from my_utils.data_preprocessing import IMG_HEIGHT, get_spectrogram_from_raw_audio
from my_utils.encoding_convertions import STEP_CHANGE_TOKEN, VOICE_CHANGE_TOKEN
from networks.crnn.model import CTCTrainedCRNN
from networks.transformer.encoder import HEIGHT_REDUCTION, WIDTH_REDUCTION
from networks.transformer.model import A2STransformer


def infer(
    audio_path: str,
    checkpoint_path: str,
    output_path: str = "",
    model_type: str = "transformer",
    device: str = "",
    file_format: str = "krn",
):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    file_format = file_format.lower()
    if file_format not in ("krn", "mxl"):
        print(f"Error: unknown file_format '{file_format}'. Choose 'krn' or 'mxl'.")
        return

    if not output_path:
        output_path = f"output.{file_format}"

    if not os.path.exists(audio_path):
        print(f"Error: audio file not found: {audio_path}")
        return

    if not os.path.exists(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading audio: {audio_path}")
    raw_audio, sr = librosa.load(audio_path, sr=None, mono=True)

    print(f"Preprocessing audio (sr={sr}, duration={len(raw_audio)/sr:.1f}s)")
    spectrogram = get_spectrogram_from_raw_audio(raw_audio, sr)

    x = np.expand_dims(spectrogram, 0)
    x = torch.from_numpy(x).float().unsqueeze(0)

    print(f"Loading model: {checkpoint_path}")
    device = torch.device(device)

    if model_type == "transformer":
        model = A2STransformer.load_from_checkpoint(checkpoint_path)
    elif model_type == "crnn":
        model = CTCTrainedCRNN.load_from_checkpoint(checkpoint_path)
    else:
        print(f"Error: unknown model_type '{model_type}'. Choose 'transformer' or 'crnn'.")
        return

    model = model.to(device)
    model.eval()

    T = x.shape[3]
    if T > model.max_audio_len:
        print(
            f"Warning: audio length ({T} frames) exceeds model max "
            f"({model.max_audio_len}), truncating"
        )
        x = x[:, :, :, : model.max_audio_len]

    xl = math.ceil(IMG_HEIGHT / HEIGHT_REDUCTION) * math.ceil(x.shape[3] / WIDTH_REDUCTION)
    xl_tensor = torch.tensor([xl], dtype=torch.int64, device=device)
    x = x.to(device)

    print(f"Running inference ({model_type})...")
    with torch.no_grad():
        if model_type == "transformer":
            decoded_tokens = _infer_transformer(model, x, xl_tensor, device)
        else:
            decoded_tokens = _infer_crnn(model, x)

    if not decoded_tokens:
        print("Warning: no tokens were decoded")
        return

    num_voices = get_number_of_voices(decoded_tokens)
    print(f"Detected {num_voices} voices, {len(decoded_tokens)} tokens")

    if file_format == "krn":
        print(f"Writing {output_path}")
        create_kern_file(output_path, decoded_tokens, num_voices)
        print("Done")
    else:
        _write_musicxml(output_path, decoded_tokens, num_voices)


@torch.no_grad()
def _infer_transformer(model, x, xl, device):
    x = model.encoder(x=x)
    x = model.pos_2d(x)
    x = x.flatten(2).permute(0, 2, 1).contiguous()

    sos = model.w2i[SOS_TOKEN]
    eos = model.w2i[EOS_TOKEN]

    y_in = torch.full((1, 1), sos, dtype=torch.long, device=device)
    decoded_tokens = []

    for _ in range(model.max_seq_len):
        y_out_hat = model.decoder(tgt=y_in, memory=x, memory_len=xl)
        next_tok = y_out_hat[0, :, -1].argmax(dim=-1).item()

        if next_tok == eos:
            break
        decoded_tokens.append(model.i2w[next_tok])

        y_in = torch.cat([y_in, torch.tensor([[next_tok]], device=device)], dim=1)

    return decoded_tokens


@torch.no_grad()
def _infer_crnn(model, x, device):
    yhat = model.forward(x)[0]
    yhat = yhat.log_softmax(dim=-1).detach().cpu()
    return model.ctc_greedy_decoder(yhat, model.i2w)


def get_number_of_voices(kern):
    num_voices = 0
    for token in kern:
        if token == VOICE_CHANGE_TOKEN:
            continue
        if token == STEP_CHANGE_TOKEN:
            break
        num_voices += 1
    return num_voices


def create_kern_file(out_file, kern, num_voices):
    with open(out_file, "w") as fout:
        fout.write("\t".join(["**kern"] * num_voices) + "\n")
        line = []
        for token in kern:
            if token == STEP_CHANGE_TOKEN:
                if line:
                    while len(line) < num_voices:
                        line.append(".")
                    fout.write("\t".join(line) + "\n")
                line = []
            elif token == VOICE_CHANGE_TOKEN:
                pass
            elif token == "DOT":
                line.append(".")
            else:
                line.append(token)
        if line:
            while len(line) < num_voices:
                line.append(".")
            fout.write("\t".join(line) + "\n")
        fout.write("\t".join(["*-"] * num_voices) + "\n")


def _write_musicxml(output_path, decoded_tokens, num_voices):
    try:
        from music21 import converter
    except ImportError:
        print("Error: music21 is required for MusicXML output. Install it with: pip install music21")
        return

    tmp_krn = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".krn", delete=False, encoding="utf-8"
        ) as f:
            create_kern_file(f.name, decoded_tokens, num_voices)
            tmp_krn = f.name

        print(f"Converting to MusicXML: {output_path}")
        score = converter.parse(tmp_krn)
        score.write("musicxml", fp=output_path)
        print("Done")
    finally:
        if tmp_krn and os.path.exists(tmp_krn):
            os.remove(tmp_krn)


if __name__ == "__main__":
    fire.Fire(infer)
