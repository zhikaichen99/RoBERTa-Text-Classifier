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
    parser.add_argument('--max_seq_length', type = int, default = 128)
    parser.add_argument('--freeze_bert_layer', type = eval, default = False)
    parser.add_argument('--epochs', type = int, default = 1)
    parser.add_argument('--learning_rate', type = float, default = 0.01)
    parser.add_argument('--train_batch_size', type = int, default = 64)
    parser.add_argument('--train_steps_per_epoch', type = int, default = 50)
    parser.add_argument('--validation_batch_size', type = int, default = 64)
    parser.add_argument('--validation_steps_per_epoch', type = int, default = 64)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--run_validation', type = eval, default = False)

    # Container environment
    parser.add_argument('--model_dir', type = str, default = os.environ['SM_MODEL_DIR'])
    parser.add_argument('--validation_data', type = str, default = os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--train_data', type = str, default = os.environ['SM_CHANNEL_TRAIN'])
    
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
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = args.learning_rate)

    train_correct = 0
    train_total = 0

    # Looping over each epoch
    for epoch in range(args.epoch):
        # Loop through each batch
        for i, (sent,label) in enumerate(train_data_loader):
            if i < args.train_steps_per_epoch:
                model.train()
                # Clear any previously calculated gradients
                optimizer.zero_grad()
                sent = sent.squeeze(0)
                if torch.cuda.is_available():
                    sent = sent.cuda()
                    label = label.cuda()
                # Get output
                output = model(sent)[0]
                _, predicted = torch.max(output,1)

                # calculate loss function
                loss = loss_function(output, label)
                loss.backward()
                # Update optimizer
                optimizer.step()

                if args.run_validation and i % args.validation_steps_per_epoch == 0:
                    # Run validation
                    correct = 0
                    total = 0
                    model.eval()

                    for sent, label in validation_data_loader:
                        sent = sent.squeeze(0)
                        if torch.cuda.is_available():
                            sent = sent.cuda()
                            label = label.cuda()
                        output = model(sent)[0]
                        _, predicted = torch.max(output.data, 1)

                        total = total + label.size(0)
                        correct = correct + (predicted.cpu() == label.cpu()).sum()

                    accuracy = (correct.numpy() / total) * 100.00
                    print('[epoch/step: {0}/{1}] val_loss: {2:.2f} - val_acc: {3:.2f}%'.format(epoch, i, loss.item(), accuracy))
            else:
                break
    return model

if __name__ == '__main__':

    # Parse Arguments
    args = parse_args()

    train = args.train_data
    validation = args.validation_data

    train_input_ids = train.input_ids.to_numpy()
    train_label_ids = train.label_ids.to_numpy()

    validation_input_ids = validation.input_ids.to_numpy()
    validation_label_ids = validation.label_ids.to_numpy()

    # Create PyTorch Dataset
    train_data = ReviewDataset(train_input_ids, train_label_ids)
    validation_data = ReviewDataset(validation_input_ids, validation_label_ids)

    train_data_loader = create_data_loader(train_data, args.train_batch_size)
    validation_data_loader = create_data_loader(validation_data, args.validation_batch_size)

    # Set up model configuration
    config = RobertaConfig.from_pretrained(
        'roberta-base',
        num_labels = 3,
        id2label = {0:-1, 1:0, 2:1},
        label2id = {-1:0, 0:1, 1:2}
        output_attention = True
    )

    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        config = config
    )

    