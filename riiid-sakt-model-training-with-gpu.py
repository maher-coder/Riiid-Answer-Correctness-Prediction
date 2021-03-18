%reset -f

import gc
import random
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import numpy as np
import pandas as pd


path = Path('/kaggle/input')
assert path.exists()

##%%time

data_types_dict = {
    'content_type_id': 'bool',
    'timestamp': 'int64',
    'user_id': 'int32', 
    'content_id': 'uint16', 
    'answered_correctly': 'uint8', 
    'prior_question_elapsed_time': 'float32', 
    'prior_question_had_explanation': 'bool'
}
target = 'answered_correctly'
train_df = dt.fread(path/'riiid-test-answer-prediction/train.csv', columns = set(data_types_dict.keys())).to_pandas()


train_df = train_df[train_df.content_type_id == False]

#Ordenanr timestamp
train_df = train_df.sort_values(['timestamp'], ascending=True).reset_index(drop = True)
del train_df['timestamp'], train_df['content_type_id']
gc.collect()

n_skill = train_df['content_id'].nunique()
print('number skills', n_skill)

group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (r['content_id'].values, r['answered_correctly'].values))

del train_df
gc.collect()

MAX_SEQ = 200
ACCEPTED_USER_CONTENT_SIZE = 4
EMBED_SIZE = 256
BATCH_SIZE = 64
DROPOUT = 0.1


class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq = 100):
        super(SAKTDataset, self).__init__()
        self.samples, self.n_skill, self.max_seq = {}, n_skill, max_seq
        
        self.user_ids = []
        for i, user_id in enumerate(group.index):
            if(i % 10000 == 0):
                print(f'Processed {i} users')
            content_id, answered_correctly = group[user_id]
            if len(content_id) >= ACCEPTED_USER_CONTENT_SIZE:
                if len(content_id) > self.max_seq:
                    total_questions = len(content_id)
                    last_pos = total_questions // self.max_seq
                    for seq in range(last_pos):
                        index = f'{user_id}_{seq}'
                        self.user_ids.append(index)
                        start = seq * self.max_seq
                        end = (seq + 1) * self.max_seq
                        self.samples[index] = (content_id[start:end], answered_correctly[start:end])
                    if len(content_id[end:]) >= ACCEPTED_USER_CONTENT_SIZE:
                        index = f'{user_id}_{last_pos + 1}'
                        self.user_ids.append(index)
                        self.samples[index] = (content_id[end:], answered_correctly[end:])
                else:
                    index = f'{user_id}'
                    self.user_ids.append(index)
                    self.samples[index] = (content_id, answered_correctly)
                
                
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        content_id, answered_correctly = self.samples[user_id]
        seq_len = len(content_id)
        
        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            content_id_seq[:] = content_id[-self.max_seq:]
            answered_correctly_seq[:] = answered_correctly[-self.max_seq:]
        else:
            content_id_seq[-seq_len:] = content_id
            answered_correctly_seq[-seq_len:] = answered_correctly
            
        target_id = content_id_seq[1:]
        label = answered_correctly_seq[1:]
        
        x = content_id_seq[:-1].copy()
        x += (answered_correctly_seq[:-1] == 1) * self.n_skill
        
        return x, target_id, label
    
    
TEST_SIZE = 0.1

train, val = train_test_split(group, test_size = TEST_SIZE)

train_dataset = SAKTDataset(train, n_skill, max_seq = MAX_SEQ)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers=8)

del train
gc.collect()

val_dataset = SAKTDataset(val, n_skill, max_seq=MAX_SEQ)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers=8)

del val
gc.collect()

sample_batch = next(iter(train_dataloader))
sample_batch[0].shape, sample_batch[1].shape, sample_batch[2].shape


class FFN(nn.Module):
    def __init__(self, state_size = 200, forward_expansion = 1, bn_size=MAX_SEQ - 1, dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = state_size
        
        self.lr1 = nn.Linear(state_size, forward_expansion * state_size)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(bn_size)
        self.lr2 = nn.Linear(forward_expansion * state_size, state_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu(self.lr1(x))
        x = self.bn(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = (np.triu(np.ones([seq_length, seq_length]), k = 1)).astype('bool')
    return torch.from_numpy(future_mask)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads = 8, dropout = DROPOUT, forward_expansion = 1):
        super(TransformerBlock, self).__init__()
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, forward_expansion = forward_expansion, dropout=dropout)
        self.layer_normal_2 = nn.LayerNorm(embed_dim)
        

    def forward(self, value, key, query, att_mask):
        att_output, att_weight = self.multi_att(value, key, query, attn_mask=att_mask)
        att_output = self.dropout(self.layer_normal(att_output + value))
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        x = self.ffn(att_output)
        x = self.dropout(self.layer_normal_2(x + att_output))
        return x.squeeze(-1), att_weight
    
class Encoder(nn.Module):
    def __init__(self, n_skill, max_seq=100, embed_dim=128, dropout = DROPOUT, forward_expansion = 1, num_layers=1, heads = 8):
        super(Encoder, self).__init__()
        self.n_skill, self.embed_dim = n_skill, embed_dim
        self.embedding = nn.Embedding(2 * n_skill + 1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq - 1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, forward_expansion = forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, question_ids):
        device = x.device
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_embedding(pos_id)
        x = self.dropout(x + pos_x)
        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = self.e_embedding(question_ids)
        e = e.permute(1, 0, 2)
        for layer in self.layers:
            att_mask = future_mask(e.size(0)).to(device)
            x, att_weight = layer(e, x, x, att_mask=att_mask)
            x = x.permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        return x, att_weight

class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq = 100, embed_dim = 128, dropout = DROPOUT, forward_expansion = 1, enc_layers=1, heads = 8):
        super(SAKTModel, self).__init__()
        self.encoder = Encoder(n_skill, max_seq, embed_dim, dropout, forward_expansion, num_layers=enc_layers)
        self.pred = nn.Linear(embed_dim, 1)
        
    def forward(self, x, question_ids):
        x, att_weight = self.encoder(x, question_ids)
        x = self.pred(x)
        return x.squeeze(-1), att_weight
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_model():
    return SAKTModel(n_skill, max_seq = MAX_SEQ, embed_dim = EMBED_SIZE, forward_expansion = 1, enc_layers = 1, heads = 8, dropout = 0.1)
model = create_model()

LR = 2e-3
EPOCHS = 12
MODEL_PATH = 'SAKT_model.pt'


def load_from_item(item):
    x = item[0].to(device).long()
    target_id = item[1].to(device).long()
    label = item[2].to(device).float()
    target_mask = (target_id != 0)
    return x, target_id, label, target_mask

def update_stats(tbar, train_loss, loss, output, label, num_corrects, num_total, labels, outs):
    train_loss.append(loss.item())
    pred = (torch.sigmoid(output) >= 0.5).long()
    num_corrects += (pred == label).sum().item()
    num_total += len(label)
    labels.extend(label.view(-1).data.cpu().numpy())
    outs.extend(output.view(-1).data.cpu().numpy())
    tbar.set_description('loss - {:.4f}'.format(loss))
    return num_corrects, num_total

def train_epoch(model, dataloader, optim, criterion, scheduler, device='cpu'):
    model.train()
    
    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []
    
    tbar = tqdm(dataloader)
    for item in tbar:
        x, target_id, label, target_mask = load_from_item(item)
        
        optim.zero_grad()
        output, _ = model(x, target_id)
        
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        scheduler.step()
        
        tbar.set_description('loss - {:.4f}'.format(loss))

def val_epoch(model, val_iterator, criterion, device='cpu'):
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(val_iterator)
    for item in tbar:
        x, target_id, label, target_mask = load_from_item(item)

        with torch.no_grad():
            output, atten_weight = model(x, target_id)
        
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        
        num_corrects, num_total = update_stats(tbar, train_loss, loss, output, label, num_corrects, num_total, labels, outs)

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(train_loss)

    return loss, acc, auc

def Training():
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, 
                                                    steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
    model.to(device)
    criterion.to(device)
    best_auc = 0.0
    for epoch in range(EPOCHS):
        train_epoch(model, train_dataloader, optimizer, criterion, scheduler, device)
        val_loss, avl_acc, val_auc = val_epoch(model, val_dataloader, criterion, device)
        print(f"Epoca - {epoch + 1}   Validation_loss - {val_loss:.3f}   Accuracy - {avl_acc:.4f}   AUC - {val_auc:.4f}")
        if best_auc < val_auc:
            print(f'Epoca - {epoch + 1} best model with Validation AUC: {val_auc}')
            best_auc = val_auc
        torch.save(model.state_dict(), MODEL_PATH)
        
if __name__ == "__main__":
    Training()
    
    LR = LR/10.
    EPOCHS = 3
    Training()
    
    # Save to pickle
    group.to_pickle('/kaggle/working/group.pkl')