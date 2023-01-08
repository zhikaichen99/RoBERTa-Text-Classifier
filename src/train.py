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



MODEL_NAME = 'model.pth'

PRE_TRAINED_MODEL_NAME = 'roberta-base'

def configure_model():
    classes = [-1, 0 ,1]
    # Initializing RoBERTA configuration
    config = RobertaConfig.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels = len(classes)

        id2label = {
            0: -1,
            1: 0,
            2: 1,
        },
        label2id = {
            -1: 0,
            0: 1,
            1: 2,
        }
    )
    config.output_attentions = True
    
    return config

def create_data_loader(data_file, batch_size):
    """
    Make batches of the dataset. Each element of the batches is a tuple that contains
    input_ids, attention_mask, and labels
    """
    df = pd.read_csv(data_file, usecols = ['input_ids', 'label_ids'])

    ds = 


def train_model(model, train_data_loader, df_train, validation_data_loader, df_validation, args):

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = args.learning_rate)

    if args.freeze_bert_layer:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    train_correct = 0
    train_total = 0

    for epoch in range(args.epochs):
        for i, (sent, label) in enumerate(train_data_loader):
            if i < args.train_steps_per_epoch:
                model.train()
                optimizer.zero_grad()
                sent = sent.squeeze(0)
                if torch.cuda.is_available():
                    sent = sent.cuda()
                    label = label.cuda()
                output = model(sent)[0]
