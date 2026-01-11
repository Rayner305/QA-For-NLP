import os
import re
import nltk
import torch
import string
import random
import collections
import numpy as np
import gradio as gr
import torch.nn as nn
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup
)

# Path Settings
os.environ['HF_HOME'] = 'D:/Library_Files'
os.environ['TORCH_HOME'] = 'D:/Library_Files'
os.environ['TRANSFORMERS_CACHE'] = 'D:/Library_Files'
#---------------------------------------------
OUTPUT_DIR = "D:/Hybrid_QA_Model"
os.makedirs(OUTPUT_DIR, exist_ok=True)
#---------------------------------------------
RERANKER_MODEL_PATH = os.path.join(OUTPUT_DIR, "reranker_model")
QA_MODEL_PATH = os.path.join(OUTPUT_DIR, "qa_model")
#---------------------------------------------
#Model Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RERANKER_MODEL_NAME = "allenai/scibert_scivocab_uncased"
QA_MODEL_NAME = "microsoft/deberta-v3-base"

#Training Settings
MAX_LEN = 384
STRIDE = 128
BATCH_SIZE = 2
GRAD_ACCUM = 16
LR_RERANKER = 2e-5
LR_QA = 1.5e-5
RERANKER_EPOCHS = 5
QA_EPOCHS = 4

#Cleaning
def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\{.*?\}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Download NLTK files
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)