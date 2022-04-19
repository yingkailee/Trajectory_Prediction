import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union
import pickle
import argparse
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")
global_step = 0
best_loss = float("inf")
np.random.seed(100)

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
                        default=5000,
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
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU to use")
    return parser.parse_args()

class LSTMDataset(Dataset):
    def __init__(self, data_dict: Dict[str, Any], args: Any, mode: str):
        pass

    def __len__(self):
        pass
        #return self.data_size

    def __getitem__(self, idx: int):
        pass

def save_checkpoint(save_dir: str, state: Dict[str, Any]) -> None:
    filename = "{}/LSTM.pth.tar".format(save_dir)
    torch.save(state, filename)

def load_checkpoint(checkpoint_file: str, model: Any, optimizer: Any) -> Tuple[int, float]:
    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            f"=> loaded checkpoint {checkpoint_file} (epoch: {epoch}, loss: {best_loss})"
        )
    else:
        print(f"=> no checkpoint found at {checkpoint_file}")

    return epoch, best_loss

def my_collate_fn(batch: List[Any]) -> List[Any]:
    _input, output, helpers = [], [], []
    for item in batch:
        _input.append(item[0])
        output.append(item[1])
        helpers.append(item[2])
    _input = torch.stack(_input)
    output = torch.stack(output)
    return [_input, output, helpers]

def init_hidden(batch_size: int, hidden_size: int, device: Any) -> Tuple[Any, Any]:
    return (torch.zeros(batch_size, hidden_size).to(device), torch.zeros(batch_size, hidden_size).to(device),)

class AttentionPredictor(nn.Module):
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):

        super(AttentionPredictor, self).__init__()
        self.hidden_size = hidden_size

        # Encoder #1:
        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

        # Encoder #2:
        self.linear2 = nn.Linear(input_size, embedding_size)
        self.lstm2 = nn.LSTMCell(embedding_size, hidden_size)

        # Predictor:
        self.linear3 = nn.Linear(2*hidden_size, embedding_size)
        self.linear4 = nn.Linear(embedding_size, 1)

    def forward(self, x1: torch.FloatTensor, x2: torch.FloatTensor) -> Any:

        # Get description of movement of car 1:
        batch_size = x1.shape[0]
        input_length = x1.shape[1]
        hidden1 = init_hidden(batch_size, self.hidden_size, device)

        for ei in range(input_length):
            input1 = x1[:, ei, :]
            embedded1 = F.relu(self.linear1(input1))
            hidden1 = self.lstm1(embedded1, hidden1)

        # Get description of movement of car 2:
        batch_size = x2.shape[0]
        input_length = x2.shape[1]
        hidden2 = init_hidden(batch_size, self.hidden_size, device)
        for ei in range(input_length):
            input2 = x2[:, ei, :]
            embedded2 = F.relu(self.linear2(input2))
            hidden2 = self.lstm2(embedded2, hidden2)

        # Predict the impact:
        x = torch.cat((hidden1[0], hidden2[0]), dim=0)
        print(x.size())
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        return x

def train(
        train_loader: Any,
        criterion: Any,
        model: Any,
        optimizer: Any
) -> None:
    args = parse_arguments()
    global global_step
    for i, (_input1, _input2, target) in enumerate(train_loader):
        _input1, _input2 = _input1.to(device), _input2.to(device)
        target = target.to(device)
        # Set to train mode
        model.train()
        # Zero the gradients
        optimizer.zero_grad()
        predictions = model(_input1, _input2)
        loss = criterion(predictions, target)
        # Backpropagate
        loss.backward()
        optimizer.step()
        global_step += 1


def validate(
        val_loader: Any,
        epoch: int,
        criterion: Any,
        model: Any,
        optimizer: Any
) -> float:
    args = parse_arguments()
    total_loss = []
    global global_step
    for i, (_input1, _input2, target) in enumerate(val_loader):
        _input1, _input2 = _input1.to(device), _input2.to(device)
        target = target.to(device)
        # Set to train mode
        model.eval()
        # Zero the gradients
        optimizer.zero_grad()
        predictions = model(_input1, _input2)
        loss = criterion(predictions, target)

        total_loss.append(loss.item())

    # Save
    val_loss = sum(total_loss) / len(total_loss)

    if val_loss <= best_loss:
        print("Saving the model.", val_loss)
        best_loss = val_loss
        save_dir = "saved_models/lstm"
        os.makedirs(save_dir, exist_ok=True)
        save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_loss": val_loss,
                "optimizer": optimizer.state_dict()
            },
        )
    return val_loss


def test(
        test_loader: Any,
        model: Any,
        optimizer: Any,
) -> Any:
    args = parse_arguments()
    total_loss = []
    global global_step
    all_predictions = []
    for i, (_input1, _input2, target) in enumerate(test_loader):
        _input1, _input2 = _input1.to(device), _input2.to(device)
        target = target.to(device)
        # Set to train mode
        model.eval()
        # Zero the gradients
        optimizer.zero_grad()
        predictions = model(_input1, _input2)
        all_predictions.append(predictions)

    all_predictions = torch.stack(all_predictions)
    all_predictions = all_predictions.detach().cpu().numpy()

    return all_predictions

def main():
    """Main."""
    global best_loss
    args = parse_arguments()

    if args.gpu >= 0:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    print(device)

    # get data here:
    data_dict = None

    # Get model
    criterion = nn.MSELoss()
    model = AttentionPredictor()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # If model_path provided, resume from saved checkpoint
    if args.model_path is not None and os.path.isfile(args.model_path):
        epoch, _ = load_checkpoint(args.model_path, model,optimizer)
        start_epoch = epoch + 1
    else:
        start_epoch = 0

    if not args.test:

        # Get PyTorch Dataset
        train_dataset = LSTMDataset(data_dict, args, "train")
        val_dataset = LSTMDataset(data_dict, args, "val")


        # Setting Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=my_collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=my_collate_fn,
        )

        print("Training begins ...")

        decrement_counter = 0

        epoch = start_epoch
        global_start_time = time.time()

        best_loss = float("inf")
        prev_loss = best_loss
        while epoch < args.end_epoch:
            start = time.time()
            train(
                train_loader,
                criterion,
                model,
                optimizer
            )
            end = time.time()

            print(
                f"Training epoch completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
            )

            epoch += 1
            if epoch % 5 == 0:
                start = time.time()
                prev_loss, decrement_counter = validate(
                    val_loader,
                    epoch,
                    criterion,

                    model,
                    optimizer
                )
                end = time.time()
                print(
                    f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                )


    else:
        # Testing here
        pass



if __name__ == "__main__":
    main()



