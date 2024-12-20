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

from data_padding import labeld_padding, notlebeled_padding
from data_batch import train2batch, not2batch
from model import BERTClassifier, ERCClassifier, DACClassifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
#model_path = "tohoku-nlp/bert-large-japanese-v2"
#model_path = "nlp-waseda/roberta-base-japanese"


tokenizer = AutoTokenizer.from_pretrained(model_path)
data_split = 4 #window size
max_len = 64
tag_size = 3 #depending on a main task
batch_size = 1
Epoch = 10
learning_rate = 5e-6


model = BERTClassifier(tag_size).to(device)


tokenizer = AutoTokenizer.from_pretrained(model_path)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


#ERC model
ERC_tagsize = 2
model_aro = ERCClassifier(ERC_tagsize).to(device)
model_aro.load_state_dict(torch.load('Path to save auxiliary task model'), strict=False)
model_cre = ERCClassifier(ERC_tagsize).to(device)
model_cre.load_state_dict(torch.load('Path to save auxiliary task model'), strict=False)
model_int = ERCClassifier(ERC_tagsize).to(device)
model_int.load_state_dict(torch.load('Path to save auxiliary task model'), strict=False)
model_ple = ERCClassifier(ERC_tagsize).to(device)
model_ple.load_state_dict(torch.load('Path to save auxiliary task model'), strict=False)
model_pos = ERCClassifier(ERC_tagsize).to(device)
model_pos.load_state_dict(torch.load('Path to save auxiliary task model'), strict=False)


#DAC model
DAC_tagsize = 9
model_dae = DACClassifier(DAC_tagsize).to(device)
model_dae.load_state_dict(torch.load('Path to save auxiliary task model'), strict=False)


index_datasets_title, index_datasets_category = labeld_padding(datarow, tokenizer, max_len)
index_datasets_text, index_datasets_aug1, index_datasets_aug2, index_datasets_utt, index_datasets_utt1, index_datasets_utt2 = notlebeled_padding(datarow, tokenizer, data_split, max_len)


#trainer
criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)
learning_alpha = 0.85

label_epoch = 1
epoch = 0
while True:
    if label_epoch <= 2:
        label_epoch += 1

        all_loss = 0
        title_batch, category_batch = train2batch(index_datasets_title, index_datasets_category, batch_size=batch_size)
        for i in range(len(title_batch)):
            
            batch_loss = 0

            model.zero_grad()

            title_tensor = torch.tensor(title_batch[i], device=device)
            category_tensor = torch.tensor(category_batch[i], device=device)
            
            out = model(title_tensor)

            batch_loss = loss_function(out, category_tensor)
            batch_loss.backward()
            optimizer.step()
            
        epoch += 1
        
        if epoch == 10:
            break

    else:
        m_sum_alpha = epoch/10
        text_batch, aug1_batch, aug2_batch, utt_batch, utt1_batch, utt2_batch = not2batch(index_datasets_text, index_datasets_aug1, index_datasets_aug2, index_datasets_utt, index_datasets_utt1, index_datasets_utt2, batch_size=batch_size)
        for i in range(len(aug1_batch)):
            
            batch_loss = 0

            aug1_tensor = torch.tensor(aug1_batch[i], device=device)
            aug2_tensor = torch.tensor(aug2_batch[i], device=device)
            text_tensor = torch.tensor(text_batch[i], device=device)

            #main
            pred1, pred2 = model(aug1_tensor), model(aug2_tensor)
            label = model(text_tensor)
            m_main1, m_main2 = criterion(pred1, label), criterion(pred2, label)

            with torch.no_grad():
                utt_tensor = [torch.tensor(utt_batch[s][i], device=device) for s in range(data_split)]
                utt1_tensor = [torch.tensor(utt1_batch[s][i], device=device) for s in range(data_split)]
                utt2_tensor = [torch.tensor(utt2_batch[s][i], device=device) for s in range(data_split)]

                m_ecr1, m_ecr2 = 0, 0
                m_dae1, m_dae2 = 0, 0

                for u_tensor, u1_tensor, u2_tensor in zip(utt_tensor, utt1_tensor, utt2_tensor):
                    #ECR
                    pred_aro1, pred_aro2 = model_aro(u1_tensor), model_aro(u2_tensor)
                    label_aro = model_aro(u_tensor)
                    m_aro1, m_aro2 = criterion(pred_aro1, label_aro), criterion(pred_aro2, label_aro)
                    pred_cre1, pred_cre2 = model_cre(u1_tensor), model_cre(u2_tensor)
                    label_cre = model_cre(u_tensor)
                    m_cre1, m_cre2 = criterion(pred_cre1, label_cre), criterion(pred_cre2, label_cre)
                    pred_int1, pred_int2 = model_int(u1_tensor), model_int(u2_tensor)
                    label_int = model_int(u_tensor)
                    m_int1, m_int2 = criterion(pred_int1, label_int), criterion(pred_int2, label_int)
                    pred_ple1, pred_ple2 = model_ple(u1_tensor), model_ple(u2_tensor)
                    label_ple = model_ple(u_tensor)
                    m_ple1, m_ple2 = criterion(pred_ple1, label_ple), criterion(pred_ple2, label_ple)
                    pred_pos1, pred_pos2 = model_pos(u1_tensor), model_pos(u2_tensor)
                    label_pos = model_pos(u_tensor)
                    m_pos1, m_pos2 = criterion(pred_pos1, label_pos), criterion(pred_pos2, label_pos)

                    m_ecr1 += m_aro1+m_cre1+m_int1+m_ple1+m_pos1
                    m_ecr2 += m_aro2+m_cre2+m_int2+m_ple2+m_pos2
                    
                    #DAE
                    pred_dae1, pred_dae2 = model_dae(u1_tensor), model_dae(u2_tensor)
                    label_dae = model_dae(u_tensor)
                    m_dae1 += criterion(pred_dae1, label_dae)
                    m_dae2 += criterion(pred_dae2, label_dae)

                m_ecr1 /= data_split
                m_ecr2 /= data_split
                m_dae1 /= data_split
                m_dae2 /= data_split

            m1 = (m_sum_alpha*m_main1) + (((1-m_sum_alpha)/2)*m_ecr1) + (((1-m_sum_alpha)/2)*m_dae1)
            m2 = (m_sum_alpha*m_main2) + (((1-m_sum_alpha)/2)*m_ecr2) + (((1-m_sum_alpha)/2)*m_dae2)

            if m1 <= m2:
                weak_tensor = torch.tensor(aug1_batch[i], device=device)
                title_tensor = torch.tensor(aug2_batch[i], device=device)
            else:
                weak_tensor = torch.tensor(aug2_batch[i], device=device)
                title_tensor = torch.tensor(aug1_batch[i], device=device)


            model.zero_grad()

            out_tensor = softmax(model(weak_tensor))
            category_value, category_tensor = torch.max(out_tensor, 1)

            if category_value >= learning_alpha:
                
                out = model(title_tensor)

                batch_loss = loss_function(out, category_tensor)
                batch_loss.backward()
                optimizer.step()

        label_epoch = 1


torch.save(model.state_dict(), 'The path to save the created model')