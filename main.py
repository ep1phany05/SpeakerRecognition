# -*- coding:utf-8 -*-
# file name: test.py
import os
import sys
import platform

import torch.nn as nn
from tqdm import tqdm, trange
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.preprocessing import label_binarize

from torch.utils.data import Dataset, DataLoader

from models.x_vector import Xvector
from utils.plda import *
from utils.setup_seed import setup_seed


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        features, label = torch.load(file_path)
        return features, label


def prepare_dataset(hparams, dataset_path, shuffle):
    dataset = AudioDataset(dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=hparams["batch_size"],
        shuffle=shuffle,
        num_workers=hparams["num_workers"],
        pin_memory=True
    )
    return dataloader


def prepare_model(hparams, device):
    m = Xvector(
        input_dim=hparams["pca_components"],
        emb_dim=hparams["emb_dim"],
        num_classes=hparams["n_classes"],
    ).to(device)
    optim = torch.optim.AdamW(m.parameters(), lr=hparams["lr_start"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=hparams["number_of_epochs"])
    loss_fn = nn.CrossEntropyLoss()
    return m, optim, sched, loss_fn


def train_xvector(hparams, device):
    train_dataloader = prepare_dataset(hparams, hparams["processed_train"], True)
    valid_dataloader = prepare_dataset(hparams, hparams["processed_valid"], False)
    model, optimizer, scheduler, criterion = prepare_model(hparams, device)
    
    # Training loop
    best_acc = 0.0
    os.makedirs(hparams['save_folder'], exist_ok=True)
    for epoch in range(hparams['number_of_epochs']):
        # Training phase
        model.train()
        full_preds = []
        full_gts = []
        running_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                    desc=f"[Epoch {epoch + 1}/{hparams['number_of_epochs']}] Train")
        for i, train_batch in pbar:
            inputs = train_batch[0].to(device)
            labels = train_batch[1].to(device)
            if current_system == 'Windows':
                labels = labels.long()

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update progress bar description
            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, 1)
            # running_corrects = torch.sum(preds == labels.data)
            pbar.set_description(
                f"[Epoch {epoch + 1}/{hparams['number_of_epochs']}] "
                f"Training Loss: {running_loss / ((i + 1) * inputs.size(0)):.3f}")
            for pred in preds.detach().cpu().numpy():
                full_preds.append(pred)
            for lab in labels.detach().cpu().numpy():
                full_gts.append(lab)

        mean_acc = accuracy_score(full_gts, full_preds)
        mean_loss = running_loss / len(train_dataloader.dataset)
        print(f'Train_loss {mean_loss:.3f} and Train_acc {mean_acc * 100:.2f}%')

        # Valid phase
        model.eval()
        with torch.no_grad():
            full_preds = []
            full_gts = []
            running_loss = 0.0
            pbar = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), desc="Valid")
            for _, val_batch in pbar:
                inputs = val_batch[0].to(device)
                labels = val_batch[1].to(device)
                if current_system == 'Windows':
                    labels = labels.long()

                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1)
                # running_corrects = torch.sum(preds == labels.data)
                for pred in preds.detach().cpu().numpy():
                    full_preds.append(pred)
                for lab in labels.detach().cpu().numpy():
                    full_gts.append(lab)

            mean_acc = accuracy_score(full_gts, full_preds)
            mean_loss = running_loss / len(valid_dataloader.dataset)
            print(f'Val_loss {mean_loss:.3f} and Val_acc {mean_acc * 100:.2f}%')

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, f"{hparams['save_folder']}/best.pth")

            if epoch % hparams['save_interval'] == 0 and epoch != 0:
                torch.save(model.state_dict(), f"{hparams['save_folder']}/{str(epoch).zfill(4)}.pth")

    print('Training complete.')


def test_xvector(hparams, device):
    train_dataloader = prepare_dataset(hparams, hparams["processed_train"], False)
    test_dataloader = prepare_dataset(hparams, hparams["processed_test"], False)
    model, _, _, criterion = prepare_model(hparams, device)

    # Load the best model
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"{hparams['save_folder']}/best.pth"))
    else:
        model.load_state_dict(torch.load(f"{hparams['save_folder']}/best.pth", map_location=torch.device('cpu')))

    # Function to generate x_vectors
    def generate_x_vectors(dataloader, dataset_type):
        model.eval()
        with torch.no_grad():
            full_preds = []
            full_gts = []
            x_vectors = []
            train_labels = []
            running_loss = 0.0
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"{dataset_type}")
            for _, batch in pbar:
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                if current_system == 'Windows':
                    labels = labels.long()

                outputs, x_vec = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, 1)
                for pred in preds.detach().cpu().numpy():
                    full_preds.append(pred)
                for lab in labels.detach().cpu().numpy():
                    full_gts.append(lab)

                if hparams["enable_plda"]:
                    for x in x_vec.cpu().numpy():
                        x_vectors.append(x)
                    for i in labels.cpu().numpy():
                        train_labels.append([i])  # make it a 2D array

            mean_acc = accuracy_score(full_gts, full_preds)
            mean_loss = running_loss / len(dataloader.dataset)
            print(f'{dataset_type}_loss {mean_loss:.3f} and {dataset_type}_acc {mean_acc * 100:.2f}%')

            # Save x-vectors and labels to a npy file
            if hparams["enable_plda"]:
                print(f"Saving x-vectors to {hparams['save_folder']}/{dataset_type}_x_vectors.npy")
                print(f"Number of x-vectors: {len(x_vectors)}")
                np.save(
                    f"{hparams['save_folder']}/{dataset_type}_x_vectors.npy",
                    {'x_vectors': np.array(x_vectors), 'labels': np.array(train_labels)},
                )
        print(f'{dataset_type} complete.')

    # Generate x_vectors for both train and test data
    generate_x_vectors(train_dataloader, "Train")
    generate_x_vectors(test_dataloader, "Test")


def train_plda(hparams):
    # Load the x-vectors
    os.makedirs(hparams['plda_folder'], exist_ok=True)
    data = np.load(f"{hparams['save_folder']}/Train_x_vectors.npy", allow_pickle=True).item()
    train_xv = np.array(data['x_vectors'], dtype=np.float64)  # float64 is required for PLDA
    train_labels = data['labels'].squeeze(1)
    segset = np.array([f"spk_{i}" for i in range(len(train_labels))])
    stat0 = np.array([[1.0]] * len(train_labels))
    s = numpy.array([None] * len(train_labels))
    with open('data/label_encoder.txt', 'r') as f:
        lines = f.readlines()
    label_encoder = {}
    for line in lines:
        x_id, label = line.strip().split(' => ')
        label_encoder[int(label)] = x_id
    modelset = np.array([label_encoder[label].replace("'", "") for label in train_labels])

    xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)

    # Train PLDA
    ranks = [50, 100, 150, 200]
    for rank in ranks:
        plda = PLDA(rank_f=rank, nb_iter=10)
        plda.plda(xvectors_stat)
        save_plda(plda, os.path.join(hparams['plda_folder'], f'plda_v2_d{rank}'))
    print('PLDA training complete.')


def test_plda(hparams):

    train_data = np.load(f"{hparams['save_folder']}/Train_x_vectors.npy", allow_pickle=True).item()
    train_xv = np.array(train_data['x_vectors'], dtype=np.float64)  # float64 is required for PLDA
    train_labels = train_data['labels'].squeeze(1)
    plda = LinearDiscriminantAnalysis()
    plda.fit(train_xv, train_labels)

    test_data = np.load(f"{hparams['save_folder']}/Test_x_vectors.npy", allow_pickle=True).item()
    test_xv = np.array(test_data['x_vectors'], dtype=np.float64)  # float64 is required for PLDA
    test_labels = test_data['labels'].squeeze(1)
    score = plda.decision_function(test_xv)
    pred_labels = np.argmax(score, axis=1)
    print(classification_report(test_labels, pred_labels))

    # calculate EER
    train_labels_bin = label_binarize(train_labels, classes=np.arange(hparams["n_classes"]))
    test_labels_bin = label_binarize(test_labels, classes=np.arange(hparams["n_classes"]))
    eer_list = []
    for i in trange(hparams["n_classes"], desc='Calculating EER'):
        plda = LinearDiscriminantAnalysis()
        plda.fit(train_xv, train_labels_bin[:, i])
        y_score = plda.decision_function(test_xv)
        fpr, tpr, thresholds = roc_curve(test_labels_bin[:, i], y_score)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_list.append(eer)
    mean_eer = np.mean(eer_list)
    print(f'Mean EER: {mean_eer * 100:.2f}%')
    print('PLDA testing complete.')


if __name__ == '__main__':

    # Ignore UserWarnings
    import logging

    logging.getLogger().setLevel(logging.ERROR)
    current_system = platform.system()
    print(f"Current system: {current_system}")

    # Reading command line arguments.
    hparams_file, _, overrides = sb.parse_arguments(sys.argv[1:])  # ["default.yaml", run_opts, None]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides)
    setup_seed(hparams["seed"])

    # Train and test the system
    if hparams["test_only"]:
        test_xvector(hparams, device)
        if hparams["enable_plda"]:
            test_plda(hparams)
    else:
        train_xvector(hparams, device)
        test_xvector(hparams, device)
        if hparams["enable_plda"]:
            test_plda(hparams)
