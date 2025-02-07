import numpy as np
import pandas as pd

from tqdm import tqdm


class DawidSkeneEM:
    def __init__(self, crowd_labels, max_iter=10, tol=1e-4):
        self.crowd_labels = crowd_labels
        self.max_iter = max_iter
        self.tol = tol

        self._encode_data_and_get_counts()
        self.initialize_parameters()

    def _encode_data_and_get_counts(self):
        """
        Encode ids of users, objects and labels to integers
        """
        self.usr_arr = self.crowd_labels.user.unique()
        self.obj_arr = self.crowd_labels.object.unique()
        self.lbl_arr = self.crowd_labels.label.unique()

        self.num_usr = len(self.usr_arr)
        self.num_obj = len(self.obj_arr)
        self.num_lbl = len(self.lbl_arr)

        self.counts = np.zeros((self.num_usr, self.num_obj, self.num_lbl))

        # Create mappings for users, objects and labels to integers
        usr2idx = {n: i for (i, n) in enumerate(self.usr_arr)}
        obj2idx = {n: i for (i, n) in enumerate(self.obj_arr)}
        lbl2idx = {n: i for (i, n) in enumerate(self.lbl_arr)}

        for ind in tqdm(self.crowd_labels.index, postfix='processing labels', leave=False):
            row = self.crowd_labels.loc[ind]

            usr_idx = usr2idx[row.user]
            obj_idx = obj2idx[row.object]
            lbl_idx = lbl2idx[row.label]

            self.counts[usr_idx, obj_idx, lbl_idx] += 1

    def initialize_parameters(self):
        self.label_prob = np.random.random((self.num_obj, self.num_lbl))
        self.label_prob /= self.label_prob.sum(axis=1, keepdims=True)

        self.pi = np.random.random((self.num_usr, self.num_lbl, self.num_lbl))  # error_rate
        self.rho = np.full(self.num_lbl, 1.0 / self.num_lbl)

    def _e_step(self):
        for i_obj in tqdm(range(self.num_obj), postfix='e_step', leave=False):
            likelihoods = np.power(self.pi, self.counts[:, [i_obj], :]).prod(0).prod(1)
            self.label_prob[i_obj] = likelihoods * self.rho
            sum_prob = self.label_prob[i_obj].sum()
            sum_prob = np.where(sum_prob == 0, 1e-9, sum_prob)
            self.label_prob[i_obj] /= sum_prob
        return self.label_prob

    def _m_step(self, posterior):
        self.rho = posterior.sum(axis=0) / posterior.sum()

        for i_lbl in tqdm(range(self.num_lbl), postfix='m_step', leave=True):
            user_error_rate = np.dot(self.label_prob[:, i_lbl], self.counts)
            sum_error_rate = np.sum(user_error_rate, axis=1, keepdims=True)
            sum_error_rate = np.where(sum_error_rate == 0, 1e-9, sum_error_rate)
            self.pi[:, i_lbl, :] = np.where(sum_error_rate == 0, sum_error_rate,
                                            user_error_rate / sum_error_rate)

    def fit(self):
        for iteration in range(self.max_iter):
            print(f"Iteration {iteration} out of {self.max_iter}\n")
            self._e_step()
            self._m_step(self.label_prob)

        return self.pi, self.rho

    def predict(self):
        labels = np.argmax(self.label_prob, axis=1)
        preds = pd.Series(
            index=pd.Index(data=self.obj_arr, name='object'),
            data=np.array(self.lbl_arr)[labels]
        )
        return preds
