"""lstm_utils.py contains utility functions for running LSTM Baselines."""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM Baselines."""
    def __init__(self, data_in, data_out):
        # Get input
        self.input_data = torch.from_numpy(np.array(data_in).astype(np.float32))
        self.output_data = torch.from_numpy(np.array(data_out).astype(np.float32))
        self.data_size = self.input_data.shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx: int
                    ) -> Tuple[Any, Any]:
        return (self.input_data[idx],self.output_data[idx])

class ModelUtils:
    """Utils for LSTM baselines."""
    def save_checkpoint(self, save_dir: str, state: Dict[str, Any]) -> None:
        """Save checkpoint file.
        
        Args:
            save_dir: Directory where model is to be saved
            state: State of the model

        """
        filename = "{}/LSTM_rollout{}.pth.tar".format(save_dir,
                                                      state["rollout_len"])
        torch.save(state, filename)

    def load_checkpoint(
            self,
            checkpoint_file: str,
            encoder: Any,
            decoder: Any,
            encoder_optimizer: Any,
            decoder_optimizer: Any,
            use_cuda: bool,
    ) -> Tuple[int, int, float]:
        """Load the checkpoint.

        Args:
            checkpoint_file: Path to checkpoint file
            encoder: Encoder model
            decoder: Decoder model 

        Returns:
            epoch: epoch when the model was saved.
            rollout_len: horizon used
            best_loss: loss when the checkpoint was saved

        """
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            rollout_len = checkpoint["rollout_len"]
            if use_cuda:
                encoder.module.load_state_dict(
                    checkpoint["encoder_state_dict"])
                decoder.module.load_state_dict(
                    checkpoint["decoder_state_dict"])

            else:
                encoder.load_state_dict(checkpoint["encoder_state_dict"])
                decoder.load_state_dict(checkpoint["decoder_state_dict"])
            encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
            decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
            print(
                f"=> loaded checkpoint {checkpoint_file} (epoch: {epoch}, loss: {best_loss})"
            )
        else:
            print(f"=> no checkpoint found at {checkpoint_file}")

        return epoch, rollout_len, best_loss

    def load_checkpoint_single(
            self,
            checkpoint_file: str,
            model: Any,
            optimizer: Any,
            use_cuda: bool,
    ) -> Tuple[int, int, float]:
        """Load the checkpoint.

        Args:
            checkpoint_file: Path to checkpoint file
            encoder: Encoder model
            decoder: Decoder model

        Returns:
            epoch: epoch when the model was saved.
            rollout_len: horizon used
            best_loss: loss when the checkpoint was saved

        """
        if os.path.isfile(checkpoint_file):
            print("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            rollout_len = checkpoint["rollout_len"]
            if use_cuda:
                model.module.load_state_dict(
                    checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                f"=> loaded checkpoint {checkpoint_file} (epoch: {epoch}, loss: {best_loss})"
            )
        else:
            print(f"=> no checkpoint found at {checkpoint_file}")

        return epoch, rollout_len, best_loss

    def my_collate_fn(self, batch: List[Any]) -> List[Any]:
        """Collate function for PyTorch DataLoader.

        Args:
            batch: Batch data

        Returns: 
            input, output and helpers in the format expected by DataLoader

        """
        _input, output = [], []

        for item in batch:
            _input.append(item[0])
            output.append(item[1])
        _input = torch.stack(_input)
        output = torch.stack(output)
        return [_input, output]

    def init_hidden(self, batch_size: int,
                    hidden_size: int, device: Any) -> Tuple[Any, Any]:
        """Get initial hidden state for LSTM.

        Args:
            batch_size: Batch size
            hidden_size: Hidden size of LSTM

        Returns:
            Initial hidden states

        """
        return (
            torch.zeros(batch_size, hidden_size).to(device),
            torch.zeros(batch_size, hidden_size).to(device),
        )
