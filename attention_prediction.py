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
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
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
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
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
    parser.add_argument("--mlp",
                        action="store_true",
                        help="Use MLP instead of LSTM")
    return parser.parse_args()

class MLP(nn.Module):
    def __init__(self,
                 input_len: int = 20,
                 output_len: int = 30):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(2 * input_len, 500, bias=True)
        print(input_len)
        print(self.linear1)
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
        #print(decoder_outputs.size(), target.size())
        loss = criterion(decoder_outputs.view(-1), target)
        # Backpropagate
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        global_step += 1
        total_loss += loss.item()
        total_num += batch_size
    #print("Train predictor performance:")
    print("Train loss:", total_loss/total_num)

def test_mlp(test_loader: Any,  model: Any, optimizer: Any, epoch: Any, criterion: Any):
    global global_step, tail_loss
    args = parse_arguments()
    total_loss = 0
    total_num = 0
    gt = []
    pred = []
    for i, (_input, target) in enumerate(test_loader):
        _input = _input.to(device)

        target = target.to(device)
        # Set to train mode
        model.eval()
        # Zero the gradients
        optimizer.zero_grad()
        # Encoder
        batch_size = _input.shape[0]
        decoder_outputs = model(_input.view(batch_size, -1))
        decoder_outputs = decoder_outputs.view(-1)
        loss = criterion(decoder_outputs, target).mean()
        gt.extend(target.detach().cpu().numpy())
        pred.extend(decoder_outputs.detach().cpu().numpy())
        total_loss += loss.item()
        total_num += batch_size
    print("\nTest predictor performance:")
    print("Average loss:", total_loss / total_num)
    '''plt.scatter(gt, pred)
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.show()'''

class EncoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden

class DecoderRNN(nn.Module):
    """Decoder Network."""
    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden):
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden

def train(train_loader: Any, epoch: int, criterion: Any, encoder: Any, decoder: Any, encoder_optimizer: Any, decoder_optimizer: Any, model_utils: ModelUtils) -> None:

    global global_step
    total_loss = 0
    total_num = 0
    for i, (_input, target) in enumerate(train_loader):

        _input = _input.to(device)
        target = target.to(device)

        # Set to train mode
        encoder.train()
        decoder.train()

        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size, encoder.hidden_size, device)

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]
        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden
        decoder_outputs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_outputs = decoder_outputs.view(-1)
        loss = criterion(decoder_outputs, target)
        # Backpropagate
        loss = loss.mean()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        global_step += 1
        total_loss += loss.item()
        total_num += batch_size
    print("Train loss:", total_loss / total_num)
    global_step += 1

def test(loader: Any, epoch: int, criterion: Any, encoder: Any, decoder: Any, encoder_optimizer: Any, decoder_optimizer: Any, model_utils: ModelUtils):
    global best_loss
    args = parse_arguments()
    total_loss = 0
    total_num = 0
    gt = []
    pred = []
    for i, (_input, target) in enumerate(loader):
        _input = _input.to(device)
        target = target.to(device)
        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]
        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size, encoder.hidden_size, device)

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)
        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]
        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden
        decoder_outputs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        decoder_outputs = decoder_outputs.view(-1)
        loss = criterion(decoder_outputs, target).mean()
        gt.extend(target.detach().cpu().numpy())
        pred.extend(decoder_outputs.detach().cpu().numpy())
        total_loss += loss.item()
        total_num += batch_size
    #print("Test predictor performance:")
    print("Average loss:", total_loss / total_num)

    '''plt.scatter(gt, pred)
    plt.xlabel("Ground truth")
    plt.ylabel("Prediction")
    plt.show()'''

def main():
    """Main."""
    global device, train_single, train_multiple, val_single, val_multiple, norm_value
    args = parse_arguments()

    if args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    print(device)
    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    model_utils = ModelUtils()


    # Get data
    # Aggregate the data:
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

    if not args.mlp:
        encoder = EncoderRNN()
        decoder = DecoderRNN(output_size=2)

        encoder.to(device)
        decoder.to(device)

        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=750, gamma=0.1)
        decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=750, gamma=0.1)
    else:
        model = MLP(input_len=20)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)

    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, _ = model_utils.load_checkpoint(
            args.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer, use_cuda=False)
        start_epoch = epoch + 1
    else:
        start_epoch = 0

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
            if args.mlp:
                train_mlp(train_loader, epoch, criterion, model, optimizer)
            else:
                train(
                    train_loader,
                    epoch,
                    criterion,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    model_utils,
                )
            end = time.time()

            #print(f"Training epoch {epoch} completed in {round((end - start) / 60.0, 2)} mins, Total time: {round((end - global_start_time) / 60.0, 2)} mins")

            epoch += 1
            if epoch % 50 == 0:
                start = time.time()
                if args.mlp:
                    test_mlp(val_loader, model, optimizer, epoch, criterion)

                else:

                    test(val_loader, epoch, criterion, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, model_utils)

                end = time.time()
                print(
                    f"Validation completed in {round((end - start) / 60.0, 2)} mins, Total time: {round((end - global_start_time) / 60.0, 2)} mins"
                )
            if args.mlp:
                scheduler.step()
            else:
                encoder_scheduler.step()
                decoder_scheduler.step()

        if args.mlp:
            # run tests for visualizations:
            for pickle_file in pickle_list:
                start_times, durations, trajectories = scan_file(pickle_file)
                vis_scenarios(start_times, durations, trajectories, model, pickle_file.split("/")[-1].split(".")[0][4:])
    else:
        pass

if __name__ == "__main__":
    args = parse_arguments()
    main()

