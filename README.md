# RoBERTa-Text-Classifier
Train and deploy fine-tuned RoBERTa model for text classification using PyTorch on Amazon Sagemaker

### Project Status: [Complete]

## Description

This module is designed to predict the sentiment of a product review. The classifier used to predict the sentiment is a variant of BERT called RoBERTa (Robustly Optimized BERT Pretraining Approach) within a PyTorch model ran as a SageMaker Training Job.

## Repository Structure

├── Feature-Processing-SageMaker.ipynb            # Feature selection and text preprocessing notebook
├── Train-RoBERTa-model-SageMaker.ipynb           # Training RoBERTa model notebook
├── Hyperparameter-Tuning-SageMaker.ipynb         # Running hyperparameter tuning job in sagemaker
├── README.md                                     # Readme file            
├── src                                           # Folder for different scripts
│   ├── config.json                               # Model configuration file
│   ├── evaluate_model_metrics.py                 # File used to evaluate model
│   ├── inference.py                              # Making inferences/predictions
│   ├── prepare-data.py                           # Script to preprocess the data for model training
│   ├── train.py                                  # Model training script
│   └── requirements.txt                          # Libraries required for this repository