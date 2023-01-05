import pandas as pd
import os

import argparse
import subprocess
import sys

from datetime import datetime
from sklearn.model_selection import train_test_split

subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==3.5.1"])

from transformers import RobertaTokenizer

columns = ['review_id', 'sentiment', 'date', 'label_ids', 'input_ids', 'review_body']

PRE_TRAINED_MODEL_NAME = 'roberta-base'

# Load roBERTa tokenize
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case = True)


# Helper functions:

# Parse Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Process')

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
    return parser.parse_args()

# Convert object data types to string
def object_to_string(dataframe):
    for column in dataframe.columns:
        if dataframe.dtypes[column] == 'object':
            dataframe[column] = dataframe[column].astype('str').astype('string')
    return dataframe

# Convert star rating into sentiment. Function will be used in processing data
def convert_to_sentiment(rating):
    if rating in {1,2}:
        return -1
    if rating == 3:
        return 0
    if rating in {4,5}:
        return 1

# Sentiment to label id
def convert_sentiment_labelid(sentiment):
    if sentiment == -1:
        return 0
    if sentiment == 0:
        return 1
    if sentiment == 1:
        return 2


# Function to convert text to required formatting
def convert_to_bert_format(text, max_seq_length):
    """
    We need to perform the following steps to our text data:
    1. Add special tokens to the start and end of each sentence.
    2. Pad & Truncate all sentences to a single constant length
    3. Differentiate real tokens from padding tokens with the 'attention mask'.
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
    df = pd.DataFrame(data = df, columns = columns)

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

    # Split data into train, test, and validation set

    # Split into training and holdout set
    holdout_size = 1 - args.train_split_percentage
    df_train, df_holdout = train_test_split(df, test_size = holdout_size, stratify = df['sentiment'])

    # Split the holdout set into test and validation set
    test_size = args.test_split_percentage / holdout_size
    df_validation, df_test = train_test_split(df_holdout, test_size = test_size, stratify = df['sentiment'])

    df_train = df_train.reset_index(drop = True)
    df_validation = df_validation.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)

    # write data to tsv file
    train_file_path = os.path.join('/opt/ml/processing/output/train', 'training_data.csv')
    validation_file_path = os.path.join('/opt/ml/processing/output/validation', 'validation_data.csv')
    test_file_path = os.path.join('/opt/ml/processing/output/test', 'test_data.csv')

    df_train.to_csv(train_file_path)
    df_validation.to_csv(validation_file_path)
    df_test.to_csv(test_file_path)

    
    

    