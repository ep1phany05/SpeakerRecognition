# -*- coding:utf-8 -*-
# file name: plda.py
import numpy as np
import torch
from speechbrain.processing.PLDA_LDA import *
from speechbrain.utils.metric_stats import EER, minDCF


def get_train_x_vec(train_xv, train_label, x_id_train):
    """
    Generate a stat object for the training x-vectors.

    Parameters
    ----------
    train_xv: ndarray
        The x-vector

    train_label: int
        The x-vectors label

    x_id_train: string
        The x-vectors unique id

    Returns
    -------
    xvectors_stat: obj
        The x-vector stat object
    """
    # Get number of train_utterances and their dimension
    N = train_xv.shape[0]
    print('N train utt:', N)

    # Define arrays necessary for special stat object
    md = ['id' + str(train_label[i]) for i in range(N)]
    modelset = np.array(md, dtype="|O")
    sg = [str(x_id_train[i]) for i in range(N)]
    segset = np.array(sg, dtype="|O")
    s = np.array([None] * N)
    stat0 = np.array([[1.0]] * N)

    # Define special stat object
    xvectors_stat = StatObject_SB(modelset=modelset, segset=segset, start=s, stop=s, stat0=stat0, stat1=train_xv)
    return xvectors_stat


def get_x_vec_stat(xv, id):
    """
    Generate a stat object for the x-vectors.

    Parameters
    ----------
    xv: ndarray
        The x-vector

    id: int
        The x-vectors unique id

    Returns
    -------
    xv_stat: obj
        The x-vector stat object
    """
    # Get number of utterances and their dimension
    N = xv.shape[0]

    # Define arrays necessary for special stat object
    sgs = [str(id[i]) for i in range(N)]
    sets = np.array(sgs, dtype="|O")
    s = np.array([None] * N)
    stat0 = np.array([[1.0]] * N)

    # Define special stat object
    xv_stat = StatObject_SB(modelset=sets, segset=sets, start=s, stop=s, stat0=stat0, stat1=xv)
    return xv_stat


def plda_scores(plda, en_stat, te_stat):
    # Define special object for plda scoring
    ndx = Ndx(models=en_stat.modelset, testsegs=te_stat.modelset)

    # PLDA Scoring
    fast_plda_scores = fast_PLDA_scoring(en_stat, te_stat, ndx, plda.mean, plda.F, plda.Sigma, p_known=0.0)
    return fast_plda_scores


def save_plda(plda, file_path_name):
    try:
        with open(file_path_name + '.pickle', 'wb') as f:
            pickle.dump(plda, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print('Error during pickling plda: ', ex)


def load_plda(file_path_name):
    try:
        with open(file_path_name, 'rb') as f:
            return pickle.load(f)
    except Exception as ex:
        print('Error during pickling plda: ', ex)


def lda(x_vec_stat, reduced_dim=2):
    lda = LDA()
    new_train_obj = lda.do_lda(x_vec_stat, reduced_dim=reduced_dim)
    return new_train_obj


class plda_score_stat_object:
    def __init__(self, test_xv, segset, stat0, s, modelset):
        self.x_vec_test = test_xv
        self.x_id_test = modelset
        self.en_stat = get_x_vec_stat(self.x_vec_test, self.x_id_test)
        self.te_stat = get_x_vec_stat(self.x_vec_test, self.x_id_test)

        self.plda_scores = 0
        self.positive_scores = []
        self.negative_scores = []
        self.positive_scores_mask = []
        self.negative_scores_mask = []

        self.eer = 0
        self.eer_th = 0
        self.min_dcf = 0
        self.min_dcf_th = 0

        self.checked_xvec = []
        self.checked_label = []

    def test_plda(self, plda, veri_test_file_path):
        self.plda_scores = plda_scores(plda, self.en_stat, self.te_stat)
        self.positive_scores_mask = np.zeros_like(self.plda_scores.scoremat)
        self.negative_scores_mask = np.zeros_like(self.plda_scores.scoremat)

        checked_list = []
        for pair in open(veri_test_file_path):
            is_match = bool(int(pair.split(" ")[0].rstrip().split(".")[0].strip()))
            enrol_id = pair.split(" ")[1].strip()
            test_id = pair.split(" ")[2].strip()

            i = int(np.where(self.plda_scores.modelset == enrol_id)[0][0])
            if not enrol_id in checked_list:
                checked_list.append(enrol_id)
                self.checked_xvec.append(self.x_vec_test[self.x_id_test == enrol_id])
                self.checked_label.append(enrol_id)

            j = int(np.where(self.plda_scores.segset == test_id)[0][0])
            if not test_id in checked_list:
                checked_list.append(test_id)
                self.checked_xvec.append(self.x_vec_test[self.x_id_test == test_id])
                self.checked_label.append(test_id)

            current_score = float(self.plda_scores.scoremat[i, j])
            if is_match:
                self.positive_scores.append(current_score)
                self.positive_scores_mask[i, j] = 1
            else:
                self.negative_scores.append(current_score)
                self.negative_scores_mask[i, j] = 1

        self.checked_xvec = np.array(self.checked_xvec)
        self.checked_label = np.array(self.checked_label)

    def calc_eer_mindcf(self):
        self.eer, self.eer_th = EER(torch.tensor(self.positive_scores), torch.tensor(self.negative_scores))
        self.min_dcf, self.min_dcf_th = minDCF(
            torch.tensor(self.positive_scores), torch.tensor(self.negative_scores), p_target=0.5)
