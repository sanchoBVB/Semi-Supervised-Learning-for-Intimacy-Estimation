import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer
import numpy as np
import torch
from transformers import BertModel
from sklearn.utils import shuffle
import MeCab
import sqlite3

from data_padding import labeld_padding
from data_batch import test2batch
from model import BERTClassifier



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
#model_path = "tohoku-nlp/bert-large-japanese-v2"
#model_path = "nlp-waseda/roberta-base-japanese"


max_len = 64
tag_size = 3 #depending on a task


tokenizer = AutoTokenizer.from_pretrained(model_path)

model = BERTClassifier(tag_size).to(device)
model.load_state_dict(torch.load('Path to save main task model'), strict=False)

index_list, index_datasets_title_test, index_datasets_category_test = labeld_padding(datarow, tokenizer, max_len)

n = 0
a = 0

predicts_list = [["index", "result"]]

# 勾配自動計算OFF
with torch.no_grad():
    index_batch, title_batch, category_batch = test2batch(index_list, index_datasets_title_test, index_datasets_category_test, batch_size=1)

    for i in range(len(title_batch)):
        title_tensor = torch.tensor(title_batch[i], device=device)
        category_tensor = torch.tensor(category_batch[i], device=device).squeeze()

        out = model(title_tensor)
        _, predicts = torch.max(out, 1)

        n += 1
        if predicts.item() == category_tensor.item():
            a += 1
            predicts_list.append([index_batch[i][0], 1])
        else:
            predicts_list.append([index_batch[i][0], 0])

print("predict : ", a / n)