# -*- coding:utf-8 -*-
# file name: dataset.py
import os
import json
import random
import logging
from speechbrain.utils.data_utils import get_all_files
import torchaudio

logger = logging.getLogger(__name__)
SAMPLERATE = 16000


def prepare_dataset(
        data_folder,
        save_json_train,
        save_json_valid,
        save_json_test,
        split_ratio=None,
):
    if split_ratio is None:
        split_ratio = [80, 10, 10]  # train, valid, test
    train_folder = os.path.join(data_folder, "data_wav", "training_dir")
    test_folder = os.path.join(data_folder, "data_wav", "testing_dir")

    # List files and create manifest from list
    logger.info(f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}")
    extension = [".wav"]
    wav_list_train = get_all_files(train_folder, match_and=extension)
    wav_list_test = get_all_files(test_folder, match_and=extension)

    # Random split the signal list into train, valid, and test sets.
    data_split_train = split_sets(wav_list_train, split_ratio)

    # Creating json files
    create_json(data_split_train["train"], save_json_train, data_folder)
    create_json(data_split_train["valid"], save_json_valid, data_folder)
    create_json(wav_list_test, save_json_test, data_folder)


def create_json(wav_list, json_file, data_root):
    json_dict = {}
    for wav_file in wav_list:
        signal, _ = torchaudio.load(wav_file)
        signal = signal.permute(1, 0)  # [rate, channel]

        duration = signal.shape[0] / SAMPLERATE

        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", os.path.relpath(wav_file, data_root))

        # Getting speaker-id from utterance-id
        spk_id = path_parts[-2]

        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "spk_id": spk_id,
        }

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def split_sets(wav_list, split_ratio):
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split
