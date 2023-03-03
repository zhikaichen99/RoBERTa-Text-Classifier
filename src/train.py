import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader

from transformers import RobertaModel, RobertaConfig
from transformers import RobertaForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

import argparse
import os
import pandas as pd
import glob
import json


def parse_args():
    """
    This function parses the arguments to the script from the jupyter notebook cell.
    Input:
        None
    Output: 
        parser.parse_args()
    """
    # Create an ArgumentParser object 
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
    parser.add_argument('--output_dir', type = 'str', default = os.environ['SM_OUTPUT_DIR'])
    
    return parser.parse_args()

# Using PyTorch Dataset

# Class to create a PyTorch dataset object for reviews
class ReviewDataset(Dataset):
    # initialization the ReviewDataset object
    def __init__(self, input_ids_list, label_ids_list):
        """
        Input:
            input_ids_list: List of input_ids arrays of the reviews
            label_ids_list: List of labels for each review in input_ids_list
        """
        self.input_ids_list = input_ids_list
        self.label_ids_list = label_ids_list
    
    # Length Method. Determine how large the dataset is
    def __len__(self):
        """
        Returns the number of reviews in the dataset
        """
        return len(self.input_ids_list)

    # Get Item Method
    def __getitem__(self, item):
        """
        Gets the input_ids and label_ids for a review at a given index

        Inputs:
            item: Index of the review to be retrieved
        Output:
            input_ids_tensor: PyTorch LongTensor object of input_ids for the review
            label_ids_tensor: PyTorch Tensor object of label_ids for the review
        """

        # Convert list of input_ids into an array of PyTorch LongTensors
        input_ids = json.loads(self.input_ids_list[item])
        label_ids = self.label_ids_list[item]

        input_ids_tensor = torch.LongTensor(input_ids)
        label_ids_tensor = torch.tensor(label_ids, dtype = torch.long)

        return input_ids_tensor, label_ids_tensor

def create_data_loader(data, batch_size):
    """
    Create a PyTorch DataLoader object that batches the data 

    Inputs:
        data: The input data as a PyTorch Dataset object
        batch_size: Integer value indicating the size of the batch
    Outputs:
        DataLoader: PyTorch DataLoader object
    """

    return DataLoader(data,
                      batch_size = batch_size,
                      shuffle = True, # Shuffle the data before creating batches
                      drop_last = True # Drop the last incomplete batch
                     )

def train_model(model, train_data_loader, validation_data_loader, args):
    """
    Trains the model using the given train_data_loader and validates the model using the
    validation_data_loader.

    Inputs:
        model: PyTorch model object to be trained
        train_data_loader: PyTorch DataLoader object containing training data
        validation_data_loader: PyTorch DataLoader object containing validation data
        args: argparse.ArgumentParser() object containing the training parameters
    Outputs:
        model: trainined model
    """

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params = model.parameters(), lr = args.learning_rate)

    # Looping over each epoch
    for epoch in range(args.epoch):
        # Loop through each batch
        for i, (sent,label) in enumerate(train_data_loader):
            # check if maximum train steps per epoch as been reached
            if i < args.train_steps_per_epoch:
                # set the model to training mode
                model.train()
                # Clear any previously calculated gradients
                optimizer.zero_grad()
                # squeeze the tensor to remove the batch size of 1 dimension
                sent = sent.squeeze(0)
                # move the tensors to GPU if it is available
                if torch.cuda.is_available():
                    sent = sent.cuda()
                    label = label.cuda()
                # Get output of the model
                output = model(sent)[0]
                # Get the predicted labels
                _, predicted = torch.max(output,1)

                # calculate loss function
                loss = loss_function(output, label)
                loss.backward()
                # Update optimizer
                optimizer.step()

                # Check if validation is needed and if validation steps have been reached    
                if args.run_validation and i % args.validation_steps_per_epoch == 0:
                    # Run validation
                    correct = 0
                    total = 0
                    # set the model to evaluation mode
                    model.eval()

                    for sent, label in validation_data_loader:
                        sent = sent.squeeze(0)

                        # move tensors to GPU if it is available
                        if torch.cuda.is_available():
                            sent = sent.cuda()
                            label = label.cuda()
                        # get output of the model
                        output = model(sent)[0]
                        # get the predicted labels
                        _, predicted = torch.max(output.data, 1)

                        total = total + label.size(0)
                        correct = correct + (predicted.cpu() == label.cpu()).sum()
                    # calculate validation accuracy
                    accuracy = (correct.numpy() / total) * 100.00
                    # print the validation loss and accuracy
                    print('[epoch/step: {0}/{1}] val_loss: {2:.2f} - val_acc: {3:.2f}%'.format(epoch, i, loss.item(), accuracy))
            else:
                break
    # return the trained model
    return model

if __name__ == '__main__':

    # Parse Arguments
    args = parse_args()

    # read training and validation files
    training_file = glob.glob('{}/*.csv'.format(args.train_data))[0]
    validation_file = glob.glob('{}/*.csv'.format(args.validation_data))[0]
    
    train = pd.read_csv(training_file)
    validation = pd.read_csv(training_file)

    # extract input_ids and label_ids
    train_input_ids = train.input_ids.to_numpy()
    train_label_ids = train.label_ids.to_numpy()

    validation_input_ids = validation.input_ids.to_numpy()
    validation_label_ids = validation.label_ids.to_numpy()

    # Create PyTorch Dataset for training and validation data
    train_data = ReviewDataset(train_input_ids, train_label_ids)
    validation_data = ReviewDataset(validation_input_ids, validation_label_ids)

    # Creating DataLoaders for training and validation data
    train_data_loader = create_data_loader(train_data, args.train_batch_size)
    validation_data_loader = create_data_loader(validation_data, args.validation_batch_size)

    # Set up model configuration
    config = RobertaConfig.from_pretrained(
        'roberta-base',
        num_labels = 3,
        id2label = {0:-1, 1:0, 2:1},
        label2id = {-1:0, 0:1, 1:2},
        output_attention = True
    )

    # Load the pre-trainined model for sequence classification
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        config = config
    )

    # train the model
    model = train_model(model, train_data_loader, validation_data_loader, args)

    # Save models
    output_path = args.model_dir
    os.makedirs(output_path, exist_ok = True)
    

    # Save the transformer model
    transformer_path = '{}/transformer'.format(output_path)
    os.makedirs(transformer_path, exist_ok = True)
    model.save_pretrained(transformer_path)

    # Save model pytorch model
    MODEL_NAME = 'model.pth'
    save_path = os.path.join(output_path, MODEL_NAME)
    torch.save(model.state_dict(), save_path)