# -*- coding:utf-8 -*-
# file name: feature_extraction.py
import os
import sys
import json

import torch
import torchaudio
from torch import nn
import numpy as np
import speechbrain as sb
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from python_speech_features import mfcc, logfbank, delta
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from utils.dataset import prepare_dataset
from utils.rasta import rastaplp


def pad_audio(signal, pad_length):
    """Pad an audio signal with a certain length of silence at the beginning and end.

    :param signal: the audio signal to pad. Should be a 1D numpy array.
    :param pad_length: the length of silence to add at the beginning and end, in number of samples.
    :returns: the padded audio signal.
    """
    return np.pad(signal, pad_width=pad_length, mode='constant', constant_values=0)


def append_context_frames(features, num_context):
    """Append context frames to features matrix."""
    num_frames, num_ceps = features.shape
    padded_features = np.pad(features, ((num_context, num_context), (0, 0)), mode='constant', constant_values=0)
    context_frames = np.concatenate([padded_features[i:i + num_frames] for i in range(2 * num_context + 1)], axis=1)
    return context_frames


def get_wave_features(waveform, samplerate=16000, method='mfcc'):
    waveform = waveform.squeeze()  # Remove batch dimension

    if method == 'mfcc':
        feat = mfcc(waveform, samplerate)
    elif method == 'logfbank':
        feat = logfbank(waveform, samplerate)
    elif method == 'plp':
        # feat = plp(waveform, samplerate, nfilt=40)
        padded_signal = pad_audio(waveform.squeeze(0).numpy(), samplerate // 100)
        feat = rastaplp(padded_signal, samplerate).transpose(1, 0)  # in: [rate, ]  out: [num_frames, filters]
    else:
        raise ValueError(f'Unknown method {method}. Available methods are "mfcc", "plp", "ssc".')

    d_feat = delta(feat, 2)
    dd_feat = delta(d_feat, 2)

    all_features = np.concatenate([feat, d_feat, dd_feat], axis=1)
    all_features_with_context = append_context_frames(all_features, 5)  # Append 5 frames of context on each side

    return torch.from_numpy(all_features_with_context[np.newaxis])  # Add back batch dimension


def prepare_and_process_audio_files(hparams, dataset_type):
    # Prepare dataset
    prepare_dataset(
        data_folder=hparams["data_folder"],
        save_json_train=hparams["train_annotation"],
        save_json_valid=hparams["valid_annotation"],
        save_json_test=hparams["test_annotation"],
        split_ratio=hparams.get("split_ratio")
    )

    # Load json file
    with open(hparams[dataset_type + "_annotation"], 'r') as f:
        data = json.load(f)

    root_dir = hparams["data_folder"]
    processed_dir = os.path.join(root_dir, 'processed', dataset_type)
    os.makedirs(processed_dir, exist_ok=True)

    # Extract labels and fit label encoder
    labels = [item['spk_id'] for item in data.values()]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    # Save label encoder mapping to a text file
    with open(os.path.join(root_dir, 'label_encoder.txt'), 'w') as f:
        for i, class_ in enumerate(label_encoder.classes_):
            f.write(f"'{class_}' => {i}\n")

    # Initialize PCA
    pca = PCA(n_components=hparams['pca_components'])

    # Process and save audio data
    for idx in tqdm(range(len(data)), desc=f"Processing {dataset_type} data"):
        # Get the item by index
        item_key = list(data.keys())[idx]
        item = data[item_key]

        # Load audio
        wav_path = item['wav'].format(data_root=root_dir)
        waveform, rate = torchaudio.load(wav_path)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        # Extract features
        waveform = waveform.unsqueeze(1).permute(1, 0)  # Reshape waveform to [rate, 1]
        mfcc_features = get_wave_features(waveform, method="mfcc").float()  # [1, num_frames, features]
        layer_norm = nn.LayerNorm(mfcc_features.size()[1:])
        mfcc_features = layer_norm(mfcc_features)
        plp_features = get_wave_features(waveform, method="plp").float()  # [1, num_frames, features]
        layer_norm = nn.LayerNorm(plp_features.size()[1:])
        plp_features = layer_norm(plp_features)
        features = torch.cat([mfcc_features, plp_features], dim=2)
        features = pca.fit_transform(features.squeeze(0).detach().numpy())

        # Encode label
        label = torch.tensor(label_encoder.transform([item['spk_id']]))[0]

        # Save processed data
        processed_path = os.path.join(processed_dir, f"{item_key}.pth")
        torch.save((torch.tensor(features, dtype=waveform.dtype), label), processed_path)


if __name__ == '__main__':
    # Reading command line arguments.
    hparams_file, _, overrides = sb.parse_arguments(sys.argv[1:])  # ["train.yaml", run_opts, None]
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    prepare_and_process_audio_files(hparams, "train")
    prepare_and_process_audio_files(hparams, "valid")
    prepare_and_process_audio_files(hparams, "test")
