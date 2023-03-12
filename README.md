# RoBERTa-Text-Classifier

The goal of this project is to develop a sentiment analysis model using the BERT and PyTorch framework. The model will be trained on a dataset of customer reviews, where each review is labeled as positive, negative, or neutral sentiment.

The classifier used to predict the sentiment is a variant of BERT called RoBERTa (Robustly Optimized BERT Pretraining Approach) within a PyTorch model ran as a SageMaker Training Job.

## Project Motivation

The motivation for this project is to learn more about leveraging deep learning models for natural language processing tasks and gaining experience in using PyTorch as a deep learning framework. By implementing a BERT model for sentiment analysis, this project can provide valuable hands-on experience in developing deep learning models, fine-tuning them for specific tasks, and evaluating their performance.

## Repository Structure and File Description

```markdown
├── Setup-Dependencies.ipynb                      # Setting up dependencies
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
├── data
│   ├── Womens Clothing E-Commerce Reviews.csv    # Raw text data
```

## Installation

To run this project, the following libraries and packages must be installed:
* Pandas
* Matplotlib
* Seaborn
* Natural Language Toolkit (NLTK)
* re
* Transformers
* Scikit-Learn
* SageMaker
* Boto3
* PyTorch
* Json

## How to Interact with the Project

1. Clone the repository to your local machine using the following command:
```
git clone https://github.com/zhikaichen99/RoBERTa-Text-Classifier.git
```
2. Create an AWS Account. Here is a link to the AWS website: [link](https://aws.amazon.com/free/?trk=c8882cbf-4c23-4e67-b098-09697e14ffd9&sc_channel=ps&s_kwcid=AL!4422!3!453053794281!e!!g!!create%20aws%20account&ef_id=Cj0KCQiA6rCgBhDVARIsAK1kGPIX4m5bpEelw12AD6zzZwZcndACDROO6VXPaNUqcyRbTrRSmgo89VkaAlGwEALw_wcB:G:s&s_kwcid=AL!4422!3!453053794281!e!!g!!create%20aws%20account&all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)
3. Create a SageMaker Notebook Instance. Here is a link on how to do so: [link](https://sagemaker-workshop.com/introduction/notebook.html)
4. Once the notebook instance is created and running, open to the Jupyter homepage and upload the repository.
5. Run the code cells in the `Setup-Dependencies.ipynb` notebook.
6. Run the code cells in the `Feature-Processing-SageMaker.ipynb` notebook.
7. Run the code cells in the `Train-RoBERTa-model-SageMaker.ipynb` notebook.
8. Run the code cells in the `Hyperparameter-Tuning-SageMaker.ipynb` notebook.
