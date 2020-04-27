# -*- coding: utf-8 -*-

import os
import json
import glob
import tqdm
import logging
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


class Wafer40Processor(object):

    def __init__(self, path_to_x, path_to_y, path_to_c):

        self.path_to_x = path_to_x  # X_map_D2-1.npy
        self.path_to_y = path_to_y  # y_map_0820.npy
        self.path_to_c = path_to_c  # y_att_0820.npy

        self.data = np.load(self.path_to_x, allow_pickle=True)
        assert len(self.data.shape) == 4          # (B, H, W, C)
        self.labels = np.load(self.path_to_y, allow_pickle=True)
        assert len(self.labels.shape) == 1        # (B, )
        self.channel_info = np.load(self.path_to_c, allow_pickle=True)
        assert len(self.channel_info.shape) == 1  # (B, )

        self.unique_labels = list(np.unique(self.labels))
        self.label2idx = {l: i for i, l in enumerate(self.unique_labels)}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def process(self, n_splits=50, write_dir='./data/processed/'):

        X = [d[:, :, self.channel_info[i]] for i, d in enumerate(self.data)]
        X = np.stack(X, axis=0)        # (B, H, W)
        X = np.expand_dims(X, axis=1)  # (B, 1, H, W)
        X = X.astype(np.float32)

        y = [self.label2idx[l] for l in self.labels]
        y = np.array(y, dtype=np.int64)
        assert len(X) == len(y)

        # Hold out 10% of the data as test set
        temp_idx, test_idx = train_test_split(
            np.arange(len(y)),
            stratify=y,
            shuffle=True,
            random_state=2015010720,
            test_size=1/10,
        )

        X_temp, y_temp = X[temp_idx], y[temp_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        with tqdm.tqdm(total=n_splits, desc='Writing files', dynamic_ncols=True) as pbar:
            for i in range(n_splits):

                # Randomly take 10% of the data as validation set
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X_temp, y_temp,
                    stratify=y_temp,
                    shuffle=True,
                    random_state=2015010720 + i,
                    test_size=1/9,
                )

                # Write to .npz files
                os.makedirs(write_dir, exist_ok=True)
                np.savez(
                    os.path.join(write_dir, f"wafer40.{i:02}.npz"),
                    x_train=X_train, y_train=y_train,
                    x_valid=X_valid, y_valid=y_valid,
                    x_test=X_test, y_test=y_test,
                    label2idx=self.label2idx,
                    unique_labels=self.unique_labels
                )
                pbar.update(1)

        with tqdm.tqdm(total=n_splits, desc='Loading files', dynamic_ncols=True) as pbar:
            npzfiles = glob.glob(os.path.join(write_dir, "*.npz"), recursive=False)
            for npzfile in npzfiles:
                npzfile = np.load(npzfile, allow_pickle=True)
                for k in npzfile.files:
                    _ = npzfile[k]
                pbar.update(1)

        with open(os.path.join(write_dir, 'label2idx.json'), 'w') as fp:
            json.dump(self.label2idx, fp, indent=2)
        with open(os.path.join(write_dir, 'idx2label.json'), 'w') as fp:
            json.dump(self.idx2label, fp, indent=2)

        print(f"Train : Validation : Test = {len(X_train):,} : {len(X_valid):,} : {len(X_test):,}")


if __name__ == '__main__':

    def parse_args():
        parser = argparse.ArgumentParser("Creating random splits of labeled data", add_help=True)
        parser.add_argument('--path_to_x', type=str, default='./data/X_map_D2-1.npy')
        parser.add_argument('--path_to_y', type=str, default='./data/y_map_0820.npy')
        parser.add_argument('--path_to_c', type=str, default='./data/y_att_0820.npy')
        parser.add_argument('--n_splits', type=int, default=100)
        parser.add_argument('--write_dir', type=str, default='./data/processed/labeled/')

        return parser.parse_args()

    args = parse_args()
    processor = Wafer40Processor(args.path_to_x, args.path_to_y, args.path_to_c)
    processor.process(n_splits=args.n_splits, write_dir=args.write_dir)
