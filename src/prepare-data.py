from sklearn.model_selection import train_test_split

import pandas as pd
import argparse
import subprocess
import sys
import os
import glob
from pathlib import Path
import time

import argparse
import subprocess
import sys

from datetime import datetime

subprocess.check_call([sys.executable, "-m", "conda", "install", "-c", "pytorch", "pytorch", "-y"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])

from transformers import RobertaTokenizer

PRE_TRAINED_MODEL_NAME = 'roberta-base'

# Load roBERTa tokenize
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case = True)


# Helper functions:

# Parse Arguments
def parse_args():
    """
    This function parses the arguments to the script from the jupyter notebook cell.
    Input:
        None
    Output: 
        parser.parse_args()
    """
    # Create an ArgumentParser object 
    parser = argparse.ArgumentParser(description='Process')

    # Add the jupyter notebook cell arguments with their corresponding data types and default values
    parser.add_argument('--train-split-percentage', type=float,
        default=0.90,
    )
    parser.add_argument('--validation-split-percentage', type=float,
        default=0.05,
    )    
    parser.add_argument('--test-split-percentage', type=float,
        default=0.05,
    )
    parser.add_argument('--balance-dataset', type=eval,
        default=True
    )    
    parser.add_argument('--max-seq-length', type=int, 
        default=128
    )
    parser.add_argument('--output-data', type = str,
        default = '/opt/ml/processing/output',
    )

    # Parse the arguments and return the ArgumentParser object
    return parser.parse_args()

# Convert object data types to string
def object_to_string(dataframe):
    """
    This function converts the data types in given pandas dataframe to string data types
    
    Input:
        dataframe: the pandas dataframe we are working with
    Output:
        dataframe: the same pandas dataframe, however columns are now string types
    """
    # Loop through each column in the dataframe
    for column in dataframe.columns:
        # check if the column is of object data type
        if dataframe.dtypes[column] == 'object':
            # convert the column to string type
            dataframe[column] = dataframe[column].astype('str').astype('string')
    # return the converted dataframe
    return dataframe

# Convert star rating into sentiment. Function will be used in processing data
def convert_to_sentiment(rating):
    """
    Converts a numerica rating to a sentiment label

    Input:
        rating: A numeric rating between 1 and 5
    Output:
        int: a sentiment where -1 represents negative sentiment, 0 represents neutral, and 1 represents positive
    """
    if rating in {1,2}:
        return -1
    if rating == 3:
        return 0
    if rating in {4,5}:
        return 1

# Sentiment to label id
def convert_sentiment_labelid(sentiment):
    """
    Convert sentiment label to Integer ID

    Input:
        sentiment: Sentiment label (-1 for negative, 0 for neutral, 1 for positive)
    Output:
        int: Integer ID (0 for negative, 1 for neutral, 2 for positive)
    """
    if sentiment == -1:
        return 0
    if sentiment == 0:
        return 1
    if sentiment == 1:
        return 2


# Function to convert text to required formatting
def convert_to_bert_format(text, max_seq_length):
    """
    Converts the input text to the BERT format by encoding the text using the BERT tokenizer
    and returning the input IDs as a flattened list

    We need to perform the following steps to our text data:
    1. Add special tokens to the start and end of each sentence.
    2. Pad & Truncate all sentences to a single constant length
    3. Differentiate real tokens from padding tokens with the 'attention mask'.

    Input:
        text: Input text to be encoded
        max_seq_length: integer value indicating the longest number of tokens we want processed.
    Output:
        list[int]: Flattened list of input IDs for the encoded text in the BERT format

    """
    encode_plus = tokenizer.encode_plus(
        text,                         # Text to encode 
        add_special_tokens = True,    # Add '[CLS]' and '[SEP]' to start and end of sentence
        max_length = max_seq_length,  # Pad and truncate all sentences
        return_token_type_ids = False,
        pad_to_max_length = 'max_length',
        return_attention_mask = True, # Construct attention mask
        return_tensors = 'pt',        # Return PyTorch Tensor
        truncation = True
    )
    return encode_plus['input_ids'].flatten().tolist()



if __name__ == "__main__":
    
    args = parse_args()

    input_data_path = os.path.join('/opt/ml/processing/input/data', 'Womens Clothing E-Commerce Reviews.csv')

    df = pd.read_csv(input_data_path, index_col = 0)
    df = pd.DataFrame(data = df)

    # remove null values in dataset
    df = df.dropna()
    df = df.reset_index(drop = True)


    # Convert the star rating in the dataset into sentiments
    df['sentiment'] = df['Rating'].apply(lambda rating: convert_to_sentiment(rating))

    # Convert review into bert embedding
    df['input_ids'] = df['Review Text'].apply(lambda review: convert_to_bert_format(review, args.max_seq_length))

    # Convert the sentiments into label ids 
    df['label_ids'] = df['sentiment'].apply(lambda sentiment: convert_sentiment_labelid(sentiment))

    # Convert index into review_id
    df.reset_index(inplace =  True)
    df = df.rename(columns = {'index': 'review_id',
                             'Review Text': 'review_body'})
    
    # Keep necessary columns
    df = df[['review_id', 'sentiment', 'label_ids', 'input_ids', 'review_body']]
    df = df.reset_index(drop = True)

    # balance the dataset 
    if args.balance_dataset:
        # group the unbalanced dataset by sentiment class
        df_unbalanced_grouped_by = df.groupby('sentiment')
        df_balanced = df_unbalanced_grouped_by.apply(lambda x: x.sample(df_unbalanced_grouped_by.size().min()).reset_index(drop = True))

        df = df_balanced

    # Adding date feature into column to keep track of the data
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    df['date'] = timestamp

    # Split into training and holdout set
    holdout_size = 1 - args.train_split_percentage
    df_train, df_holdout = train_test_split(df, test_size = holdout_size, stratify = df['sentiment'])

    # Split the holdout set into test and validation set
    test_size = args.test_split_percentage / holdout_size
    df_validation, df_test = train_test_split(df_holdout, test_size = test_size, stratify = df_holdout['sentiment'])

    df_train = df_train.reset_index(drop = True)
    df_validation = df_validation.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)

    # Convert object types to string
    df_train = object_to_string(df_train)
    df_validation = object_to_string(df_validation)
    df_test = object_to_string(df_test)

    # write data to tsv file
    train_file_path = '{}/sentiment/train'.format(args.output_data)
    validation_file_path = '{}/sentiment/validation'.format(args.output_data)
    test_file_path = '{}/sentiment/test'.format(args.output_data)

    df_train.to_csv('{}/train.csv'.format(train_file_path))
    df_validation.to_csv('{}/validation.csv'.format(validation_file_path))
    df_test.to_csv('{}/test.csv'.format(test_file_path))


    
    

    