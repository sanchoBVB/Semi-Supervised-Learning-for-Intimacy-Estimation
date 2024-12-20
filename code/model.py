import torch
import torch.nn as nn
from transformers import BertModel


class BERTClassifier(nn.Module):
    def __init__(self, tagset_size, model_path=model_path):
        super(BERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_path)

        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence):
        vec, pooler_output, attentions = self.bert(sentence, output_attentions=True, return_dict=False)
        vec = vec[:,0,:]
        vec = vec.view(-1, self.bert.config.hidden_size)
        tag_space = self.hidden2tag(vec)
        tag_scores = self.softmax(tag_space)
        return tag_scores
    

class ERCClassifier(nn.Module):
    def __init__(self, tagset_size, model_path=model_path):
        super(BERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_path)

        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence):
        vec, pooler_output, attentions = self.bert(sentence, output_attentions=True, return_dict=False)
        vec = vec[:,0,:]
        vec = vec.view(-1, self.bert.config.hidden_size)
        tag_space = self.hidden2tag(vec)
        tag_scores = self.softmax(tag_space)
        return tag_scores
    

class DACClassifier(nn.Module):
    def __init__(self, tagset_size, model_path=model_path):
        super(BERTClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_path)

        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, tagset_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence):
        vec, pooler_output, attentions = self.bert(sentence, output_attentions=True, return_dict=False)
        vec = vec[:,0,:]
        vec = vec.view(-1, self.bert.config.hidden_size)
        tag_space = self.hidden2tag(vec)
        tag_scores = self.softmax(tag_space)
        return tag_scores