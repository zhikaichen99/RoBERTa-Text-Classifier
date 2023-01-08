import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoaded

from transformers import RobertaModel, RobertaConfig
from transformers import RobertaForSequenceClassification
from transformers import AdaW, get_linear_schedule_with_warmup

import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument('--max_seq_length', type = int,
        default = 128
    )

    parser.add_argument('--freeze_bert_layer', type = eval,
        default = False    
    )

    parser.add_argument('--epochs', type = int,
        deafult = 1
    )

    parser.add_argument('--learning_rate', type = float,
        default = 0.01
    )

    parser.add_argument('--train_batch_size', type = int,
        default = 64
    )

    parser.add_argument('--train_steps_per_epoch', type = int,
        default = 50
    )

    parser.add_argument('--validation_batch_size', type = int,
        default = 64
    )

    parser.add_argument('--validation_steps_per_epoch', type = int,
        default = 64
    )

    parser.add_argument('--seed', type = int,
        default = 42
    )

    parser.add_argument('--run_validation', type = eval,
        default = False
    )

    return parser.parse_args()



# Using PyTorch Dataset

class ReviewDataset(Dataset):
    # initialization function
    def __init__(self, input_ids_list, label_ids_list):
        self.input_ids_list = input_ids_list
        self.label_ids_list = label_ids_list
    
    # Length Method. Determine how large the dataset is
    def __len__(self):
        return len(self.input_ids_list)

    # Get Item Method
    def __getitem__(self, item):
        # Convert list of input_ids into an array of PyTorch LongTensors
        input_ids = json.loads(self.input_ids_list[item])
        label_ids = self.label_ids_list[item]

        input_ids_tensor = torch.LongTensor(input_ids)
        label_ids_tensor = torch.tensor(label_ids, dtype = torch.long)

        return input_ids_tensor, label_ids_tensor

def create_data_loader(data, batch_size):

    return DataLoader(data,
                      batch_size = batch_size,
                      shuffle = True,
                      drop_last = True
                     )

def train_model(model, train_data_loader, validation_data_loader, args):