import os
import numpy as np
import numpy.typing as npt
import torch

from typing import IO, Any, BinaryIO

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Returns:
    Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    inputs = []
    labels = []
    
    dataset_length = len(dataset)
    for _ in range(batch_size):
        start_index = np.random.randint(0, dataset_length - context_length)
        end_index = start_index + context_length
        
        input_seq = dataset[start_index:end_index]
        label_seq = dataset[start_index + 1:end_index + 1]
        
        inputs.append(input_seq)
        labels.append(label_seq)
    
    inputs_tensor = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    labels_tensor = torch.tensor(np.array(labels), dtype=torch.long, device=device)
    return inputs_tensor, labels_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    iteration = checkpoint["iteration"]
    return iteration