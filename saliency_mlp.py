import glob
import os
import shutil
import tempfile

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
from typing import Any, Dict, List, Tuple, Union
import pickle
import argparse
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
from random import random, randint

from lstm_utils import ModelUtils, LSTMDataset

from get_data import read_file
from get_visuals import scan_file, vis_scenarios

from collections import defaultdict

device = torch.device("cpu")
global_step = 0
best_loss = float("inf")
np.random.seed(100)
_OBS_DURATION_TIMESTEPS = 20
tail_loss = 100
val_tail_loss = 100

global train_single, train_multiple, val_single, val_multiple, norm_value

def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=512,
                        help="Test batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=512,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=512,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=200,
                        help="Last epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument(
        "--model_name",
        required=False,
        type=str,
        help=
        "name of the model to be saved.",
    )
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU to use")
    return parser.parse_args()

class MLP(nn.Module):
    def __init__(self,
                 input_len: int = 20,
                 output_len: int = 30):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2 * input_len, 500, bias=True)
        self.linear2 = nn.Linear(500, 200, bias=True)
        self.output_len = output_len
        self.linear3 = nn.Linear(200, 1, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

def train_mlp(train_loader: Any, epoch: int, criterion: Any, model: Any, optimizer: Any) -> None:
    global global_step
    args = parse_arguments()
    start = time.time()
    total_loss = 0
    total_num = 0
    for i, (_input, target) in enumerate(train_loader):
        start = time.time()
        _input = _input.to(device)
        #target = target.type(torch.LongTensor)
        target = target.to(device)
        # Set to train mode
        model.train()
        # Zero the gradients
        optimizer.zero_grad()
        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        decoder_outputs = model(_input.view(batch_size, -1))
        loss = criterion(decoder_outputs.view(-1), target)
        # Backpropagate
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        global_step += 1
        total_loss += loss.item()
        total_num += batch_size
    print("Train predictor performance:")
    print("Train loss:", total_loss/total_num)

def test_mlp(test_loader: Any, model: Any, optimizer: Any, criterion: Any):
    global global_step, tail_loss
    
    total_loss, total_num = 0, 0
    gt, pred = [], []

    count = 0
    for i, (_input, target) in enumerate(test_loader):
        count += 1
        #print(_input)
        #print(len(_input))
        #print(type(_input)
        #print(_input[1])
        #print(type(_input[1]))
        #print(_input[0][0])
        #print(type(_input[0][0]))
        
        _input = _input.to(device)
        target = target.to(device)
        
        model.eval()
        # Zero the gradients
        optimizer.zero_grad()
        # Encoder
        batch_size = _input.shape[0]
        decoder_outputs = model(_input.view(batch_size, -1))
        decoder_outputs = decoder_outputs.view(-1)

        #print(len(decoder_outputs))

        loss = criterion(decoder_outputs, target).mean()
        gt.extend(target.detach().cpu().numpy())
        pred.extend(decoder_outputs.detach().cpu().numpy())
        total_loss += loss.item()
        total_num += batch_size

    print("qWLERH!L@J#$", count)

    print("\nTest predictor performance:")
    print("Average loss:", total_loss / total_num)

    '''plt.scatter(gt, pred)
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.show()'''

def infer_mlp(_input, model):
    _input = _input.to(device)
    model.eval()
    batch_size = _input.shape[0]
    outputs = model(_input.view(batch_size, -1))
    outputs = outputs.view(-1)

    return outputs

def main():
    global device, train_single, train_multiple, val_single, val_multiple, norm_value

    args = parse_arguments()

    if args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
 
    model_utils = ModelUtils()

    # Data Aggregation
    training_files_dir = "./data/"
    pickle_list = glob.glob(training_files_dir+"*")
    data_in = np.empty((0, 20, 2))
    
    data_out = np.empty(0)
    print("All files found:", pickle_list)
    for pickle_file in pickle_list:
        print("Reading file:", pickle_file)
        data, labels = read_file(pickle_file, show=False)
        data_in = np.vstack((data_in, data))
        
        data_out = np.concatenate((data_out, labels))
        #print(data_in.shape)
        #print(data_in)
        break # TRY ONLY 1 FILE FOR NOW

    indices_to_keep = np.argwhere(data_out>= 0.01).ravel()
    data_in = data_in[indices_to_keep, :]
    data_out = data_out[indices_to_keep]
    data_out *= 100

    # Split the data:
    indices = np.arange(len(data_in))
    np.random.shuffle(indices)
    n = int(0.7*len(indices))
    train_data_in, train_data_out = data_in[:n], data_out[:n]
    test_data_in, test_data_out = data_in[n:], data_out[n:]

    print("Data size:", "Training:", len(train_data_in), "Test:", len(test_data_in))

    # Get model
    criterion = nn.MSELoss()

    model = MLP(input_len=20)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)

    start_epoch = 0
    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, _ = model_utils.load_checkpoint(
            args.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer, use_cuda=False)
        start_epoch = epoch + 1

    if not args.test:
        # Get PyTorch Dataset
        train_dataset = LSTMDataset(train_data_in, train_data_out)
        val_dataset = LSTMDataset(test_data_in, test_data_out)

        epsilon = 0.001
        train_weights = (train_data_out+epsilon)
        #train_weights = np.ones_like(train_weights)
        train_weights /= np.sum(train_weights)

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))

        # Setting Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            sampler = train_sampler,
            drop_last=False,
            collate_fn=model_utils.my_collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn,
        )

        print("Training begins ...")

        epoch = start_epoch
        global_start_time = time.time()

        best_loss = float("inf")
        prev_loss = best_loss
        while epoch < args.end_epoch:
            start = time.time()
            train_mlp(train_loader, epoch, criterion, model, optimizer)
            end = time.time()

            #print(f"Training epoch {epoch} completed in {round((end - start) / 60.0, 2)} mins, Total time: {round((end - global_start_time) / 60.0, 2)} mins")

            epoch += 1
            if epoch % 50 == 0:
                start = time.time()

                test_mlp(val_loader, model, optimizer, criterion)

                end = time.time()
                print(
                    f"Validation completed in {round((end - start) / 60.0, 2)} mins, Total time: {round((end - global_start_time) / 60.0, 2)} mins"
                )
            scheduler.step()

        save_dir = "saved_models"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), save_dir + "/saved_mlp.pkl")

        # ERROR FROM VISUALIZATIONS MLP, COMMENTING OUT FOR NOW
        #if args.mlp:
        #    for pickle_file in pickle_list:
        #        start_times, durations, trajectories = scan_file(pickle_file)
        #        vis_scenarios(start_times, durations, trajectories, model, pickle_file.split("/")[-1].split(".")[0][4:])
    else:
        model.load_state_dict(torch.load("saved_models/saved_mlp.pkl"))

        f = open("forecasting_sample/data/2645.csv", "r")
        print("FILE HEADER: ", f.readline())

        # car: [(time, (x, y))]
        trajs = defaultdict(list)
        try:
            while True:
                split_line = f.readline().split(",")
                car_id, cur_time, x, y = split_line[1], split_line[0], split_line[3], split_line[4]
                trajs[car_id].append((cur_time, (x, y)))
        except:
            f.close()

        # assume AV lasts longer than 20
        filtered_trajs = {}
        main = "00000000-0000-0000-0000-000000000000"
        beg, end = trajs[main][0][0], trajs[main][19][0]
        
        for car, traj in trajs.items():
            if traj[0][0] <= beg and traj[-1][0] >= end:
                filtered_trajs[car] = traj[:20]

        nested_trajs = np.empty((len(filtered_trajs), 20, 2))
        ctr = 0
        for traj in filtered_trajs.values():
            xy_traj = []

            for _, xy in traj:
                pair = np.array([xy[0], xy[1]])
                xy_traj.append(pair)
            xy_traj = np.array(xy_traj)
            nested_trajs[ctr] = xy_traj

            if ctr > 0:
                nested_trajs[ctr] -= nested_trajs[0]

            ctr += 1
        
        outputs = infer_mlp(torch.from_numpy(np.array(nested_trajs).astype(np.float32)), model)

if __name__ == "__main__":
    args = parse_arguments()
    main()
