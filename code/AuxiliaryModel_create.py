import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
from sklearn.utils import shuffle
import MeCab
import sqlite3

from data_padding import labeld_padding
from data_batch import train2batch
from model import BERTClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
#model_path = "tohoku-nlp/bert-large-japanese-v2"
#model_path = "nlp-waseda/roberta-base-japanese"


max_len = 64
tag_size = 9 #depending on an auxiliary task
batch_size = 16
Epoch = 10
learning_rate = 5e-6


model = BERTClassifier(tag_size).to(device)


tokenizer = AutoTokenizer.from_pretrained(model_path)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


index_datasets_title, index_datasets_category = labeld_padding(datarow, tokenizer, max_len)


for epoch in range(Epoch):
    all_loss = 0
    title_batch, category_batch = train2batch(index_datasets_title, index_datasets_category, batch_size=batch_size)
    for i in range(len(title_batch)):
        if len(title_batch[i]) < batch_size:
            continue
        batch_loss = 0

        model.zero_grad()

        title_tensor = torch.tensor(title_batch[i], device=device)
        category_tensor = torch.tensor(category_batch[i], device=device)

        out = model(title_tensor)

        batch_loss = loss_function(out, category_tensor)
        batch_loss.backward()
        optimizer.step()


torch.save(model.state_dict(), 'The path to save the created model')