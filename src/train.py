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