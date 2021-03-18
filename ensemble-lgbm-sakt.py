%reset -f

import riiideducation
import pandas as pd
import numpy as np
import gc
from collections import defaultdict
from tqdm.notebook import tqdm
import pickle
import lightgbm as lgb

import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import psutil


####
MAX_SEQ = 100
####


dtype = {'timestamp':'int64', 
         'user_id':'int32' ,
         'content_id':'int16',
         'content_type_id':'int8',
         'answered_correctly':'int8'}

train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', usecols=[1, 2, 3, 4, 7], dtype=dtype)
train_df.head()


train_df = train_df[train_df.content_type_id == False]

#ordenar por timestamp
train_df = train_df.sort_values(['timestamp'], ascending=True).reset_index(drop = True)


skills = train_df["content_id"].unique()
n_skill = len(skills)

group = train_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
            r['content_id'].values,
            r['answered_correctly'].values))

del train_df
gc.collect()


import random
random.seed(1)


class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ): ####### 100
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = group
        
        self.user_ids = []
        for user_id in group.index:
            q, qa = group[user_id]
            if len(q) < 2: ####### 10
                continue
            self.user_ids.append(user_id)
            

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        
        if seq_len >= self.max_seq:
            
            if random.random()>0.1:
                start = random.randint(0,(seq_len-self.max_seq))
                end = start + self.max_seq
                q[:] = q_[start:end]
                qa[:] = qa_[start:end]
            else:
                
                q[:] = q_[-self.max_seq:]
                qa[:] = qa_[-self.max_seq:]
        else:
            
            if random.random()>0.1:
                
                start = 0
                end = random.randint(2,seq_len)
                seq_len = end - start
                q[-seq_len:] = q_[0:seq_len]
                qa[-seq_len:] = qa_[0:seq_len]
            else:
                
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_

        
        target_id = q[1:]
        label = qa[1:]

        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x, target_id, label
    


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, n_skill, max_seq=MAX_SEQ, embed_dim=128): ####### 100->MAX_SEQ
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, x, question_ids):
        device = x.device        
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)

        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        att_output = self.layer_normal(att_output + e)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight
    


def train_epoch(model, train_iterator, optim, criterion, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        x = item[0].to(device).long()
        target_id = item[1].to(device).long()
        label = item[2].to(device).float()

        optim.zero_grad()
        output, atten_weight = model(x, target_id)
        loss = criterion(output, label)
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        output = output[:, -1]
        label = label[:, -1] 
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc

class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, max_seq=MAX_SEQ): ####### 100
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["user_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["user_id"]
        target_id = test_info["content_id"]

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_, qa_ = self.samples[user_id]
            
            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_          
        
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill
        
        questions = np.append(q[2:], [target_id])
        
        return x, questions
    
for user_id in group.index:
    q, qa = group[user_id]
    if len(q)>MAX_SEQ:
        group[user_id] = (q[-MAX_SEQ:],qa[-MAX_SEQ:])
        

pickle.dump(group, open("group.pkl", "wb"))
del group
gc.collect()


def añadir_features(df, features_dicts):
    
    #Usuario
    tasa_acierto_usuario = np.zeros(len(df), dtype = np.float32)
    elapsed_time_u_avg = np.zeros(len(df), dtype = np.float32)
    explanation_u_avg = np.zeros(len(df), dtype = np.float32)
    user_pause_timestamp_1 = np.zeros(len(df), dtype = np.float32)
    user_pause_timestamp_2 = np.zeros(len(df), dtype = np.float32)
    user_pause_timestamp_3 = np.zeros(len(df), dtype = np.float32)
    user_pause_timestamp_incorrect = np.zeros(len(df), dtype = np.float32)
    cont_preguntas_corr_user_f = np.zeros(len(df), dtype = np.int32)
    cont_preguntas_user_f = np.zeros(len(df), dtype = np.int32)
    CUMULATIVE_ELO_USER = np.zeros(len(df), dtype = np.int32)
    # -----------------------------------------------------------------------
    # Question features
    tasa_acierto_pregunta = np.zeros(len(df), dtype = np.float32)
    elapsed_time_q_avg = np.zeros(len(df), dtype = np.float32)
    explanation_q_avg = np.zeros(len(df), dtype = np.float32)
    # -----------------------------------------------------------------------
    # User Question
    intentos = np.zeros(len(df), dtype = np.int8)
    
    for num, row in enumerate(tqdm(df[['user_id', 'content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','mean_question_accuracy']].itertuples(), total=df.shape[0])):
        
        #ELO
        CUMULATIVE_ELO_USER[num] = features_dicts['CUMULATIVE_ELO_USER'][row.user_id]
        
        # User features
        # ------------------------------------------------------------------
        if features_dicts['cont_preguntas_user'][row.user_id] != 0:
            tasa_acierto_usuario[num] = features_dicts['cont_preguntas_corr_user'][row.user_id] / features_dicts['cont_preguntas_user'][row.user_id]
            elapsed_time_u_avg[num] = features_dicts['elapsed_time_u_sum'][row.user_id] / features_dicts['cont_preguntas_user'][row.user_id]
            explanation_u_avg[num] = features_dicts['explanation_u_sum'][row.user_id] / features_dicts['cont_preguntas_user'][row.user_id]
            cont_preguntas_corr_user_f[num] = features_dicts['cont_preguntas_corr_user'][row.user_id]
            cont_preguntas_user_f[num] = features_dicts['cont_preguntas_user'][row.user_id]
            
        else:
            tasa_acierto_usuario[num] = np.nan
            elapsed_time_u_avg[num] = np.nan
            explanation_u_avg[num] = np.nan
            cont_preguntas_corr_user_f[num] = 0
            cont_preguntas_user_f[num] = 0

            
        if len(features_dicts['timestamp_u'][row.user_id]) == 0:
            user_pause_timestamp_1[num] = np.nan
            user_pause_timestamp_2[num] = np.nan
            user_pause_timestamp_3[num] = np.nan
        elif len(features_dicts['timestamp_u'][row.user_id]) == 1:
            user_pause_timestamp_1[num] = row.timestamp - features_dicts['timestamp_u'][row.user_id][0]
            user_pause_timestamp_2[num] = np.nan
            user_pause_timestamp_3[num] = np.nan
        elif len(features_dicts['timestamp_u'][row.user_id]) == 2:
            user_pause_timestamp_1[num] = row.timestamp - features_dicts['timestamp_u'][row.user_id][1]
            user_pause_timestamp_2[num] = row.timestamp - features_dicts['timestamp_u'][row.user_id][0]
            user_pause_timestamp_3[num] = np.nan
        elif len(features_dicts['timestamp_u'][row.user_id]) == 3:
            user_pause_timestamp_1[num] = row.timestamp - features_dicts['timestamp_u'][row.user_id][2]
            user_pause_timestamp_2[num] = row.timestamp - features_dicts['timestamp_u'][row.user_id][1]
            user_pause_timestamp_3[num] = row.timestamp - features_dicts['timestamp_u'][row.user_id][0]
        
        user_pause_timestamp_incorrect[num] = row.timestamp - features_dicts['timestamp_u_incorrect'][row.user_id]
          
        # ------------------------------------------------------------------
        # Question features assignation
        if features_dicts['cont_preguntas'][row.content_id] != 0:
            tasa_acierto_pregunta[num] = features_dicts['cont_preguntas_corr'][row.content_id] / features_dicts['cont_preguntas'][row.content_id]
            elapsed_time_q_avg[num] = features_dicts['elapsed_time_q_sum'][row.content_id] / features_dicts['cont_preguntas'][row.content_id]
            explanation_q_avg[num] = features_dicts['explanation_q_sum'][row.content_id] / features_dicts['cont_preguntas'][row.content_id]
        else:
            tasa_acierto_pregunta[num] = np.nan
            elapsed_time_q_avg[num] = np.nan
            explanation_q_avg[num] = np.nan
        # ------------------------------------------------------------------
        # User Question assignation
        intentos[num] = features_dicts['intentos_dict'][row.user_id][row.content_id]
        
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Actualizaciones
        features_dicts['cont_preguntas_user'][row.user_id] += 1
        features_dicts['elapsed_time_u_sum'][row.user_id] += row.prior_question_elapsed_time
        features_dicts['explanation_u_sum'][row.user_id] += row.prior_question_had_explanation
        if len(features_dicts['timestamp_u'][row.user_id]) == 3:
            features_dicts['timestamp_u'][row.user_id].pop(0)
            features_dicts['timestamp_u'][row.user_id].append(row.timestamp)
        else:
            features_dicts['timestamp_u'][row.user_id].append(row.timestamp)
        # ------------------------------------------------------------------
        # Question features updates
        features_dicts['cont_preguntas'][row.content_id] += 1
        features_dicts['elapsed_time_q_sum'][row.content_id] += row.prior_question_elapsed_time
        features_dicts['explanation_q_sum'][row.content_id] += row.prior_question_had_explanation
        # ------------------------------------------------------------------
        #User Question updates
        features_dicts['intentos_dict'][row.user_id][row.content_id] += 1
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------

        #Actualizacion ELO
        features_dicts['CUMULATIVE_ELO_USER'][row.user_id] += (1 - row.mean_question_accuracy)*100
    
    user_df = pd.DataFrame({'%_acierto_usuario': tasa_acierto_usuario, 'elapsed_time_u_avg': elapsed_time_u_avg, 'explanation_u_avg': explanation_u_avg, 
                            '%_acierto_pregunta_CONT': tasa_acierto_pregunta, 'elapsed_time_q_avg': elapsed_time_q_avg, 'explanation_q_avg': explanation_q_avg, 
                            'intentos': intentos, 'user_pause_timestamp_1': user_pause_timestamp_1, 'user_pause_timestamp_2': user_pause_timestamp_2,
                            'user_pause_timestamp_3': user_pause_timestamp_3, 'user_pause_timestamp_incorrect': user_pause_timestamp_incorrect,
                            'cont_preguntas_corr_user':cont_preguntas_corr_user_f, 'cont_preguntas_user': cont_preguntas_user_f, 'CUMULATIVE_ELO_USER':CUMULATIVE_ELO_USER}) 
    
    
    del tasa_acierto_usuario, cont_preguntas_user_f,CUMULATIVE_ELO_USER, cont_preguntas_corr_user_f, elapsed_time_u_avg, explanation_u_avg, tasa_acierto_pregunta, elapsed_time_q_avg, explanation_q_avg, intentos, user_pause_timestamp_1, user_pause_timestamp_2,user_pause_timestamp_3, user_pause_timestamp_incorrect
 
    df = pd.concat([df, user_df], axis = 1)
    del user_df
    
    #Features extra
    df['correction'] = df['user_pause_timestamp_1'] / df['user_pause_timestamp_incorrect'] + df['prior_question_had_explanation'] + df['intentos']
    df['user_pause_timestamp_ratio_1'] = df['user_pause_timestamp_1'] / df['user_pause_timestamp_2']
    df['%_media_armonica'] = 2*df['%_acierto_usuario']*df['mean_question_accuracy']/(df['%_acierto_usuario'] + df['mean_question_accuracy'])
    df['%_media_armonica'].fillna(0.642673913, inplace = True)
    df['user_pause_timestamp_MEAN'] = (df['user_pause_timestamp_1'] + df['user_pause_timestamp_2'] + df['user_pause_timestamp_3'])/3
    df['user_pause_timestamp_MEAN_RATIO'] = df['user_pause_timestamp_1']/df['user_pause_timestamp_MEAN']
    df['ELO'] = (df['CUMULATIVE_ELO_USER'] + 4*df['user_pause_timestamp_MEAN_RATIO']*df['cont_preguntas_corr_user'] - 4*df['user_pause_timestamp_MEAN_RATIO']*(df['cont_preguntas_user']-df['cont_preguntas_corr_user']))/df['cont_preguntas_user']
    df.replace(np.inf, 0, inplace = True)
    df[['ELO','correction','user_pause_timestamp_ratio_1','user_pause_timestamp_MEAN','%_media_armonica','user_pause_timestamp_MEAN_RATIO']] = df[['ELO','correction','user_pause_timestamp_ratio_1','user_pause_timestamp_MEAN','%_media_armonica','user_pause_timestamp_MEAN_RATIO']].astype(np.float32)
    
    return df

def actualizar_features_inicio(df, features_dicts):
    
    for num, row in enumerate(tqdm(df[['user_id', 'answered_correctly', 'content_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'timestamp','mean_question_accuracy']].itertuples(), total=df.shape[0])):
        features_dicts['cont_preguntas_user'][row.user_id] += 1
        features_dicts['elapsed_time_u_sum'][row.user_id] += row.prior_question_elapsed_time
        features_dicts['explanation_u_sum'][row.user_id] += row.prior_question_had_explanation
        if len(features_dicts['timestamp_u'][row.user_id]) == 3:
            features_dicts['timestamp_u'][row.user_id].pop(0)
            features_dicts['timestamp_u'][row.user_id].append(row.timestamp)
        else:
            features_dicts['timestamp_u'][row.user_id].append(row.timestamp)
        # ------------------------------------------------------------------
        # Question features updates
        features_dicts['cont_preguntas'][row.content_id] += 1
        features_dicts['elapsed_time_q_sum'][row.content_id] += row.prior_question_elapsed_time
        features_dicts['explanation_q_sum'][row.content_id] += row.prior_question_had_explanation
        # ------------------------------------------------------------------
        # User Question updates
        features_dicts['intentos_dict'][row.user_id][row.content_id] += 1
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # User features updates
        features_dicts['cont_preguntas_corr_user'][row.user_id] += row.answered_correctly
        if row.answered_correctly == 0:
            features_dicts['timestamp_u_incorrect'][row.user_id] = row.timestamp
        # ------------------------------------------------------------------
        # Question features updates
        features_dicts['cont_preguntas_corr'][row.content_id] += row.answered_correctly
        #actualizacion ELO
        features_dicts['CUMULATIVE_ELO_USER'][row.user_id] += (1 - row.mean_question_accuracy)*100
        
def actualizar_features(df, features_dicts):
    
    for row in tqdm(df[['user_id', 'answered_correctly', 'content_id', 'timestamp']].itertuples(), total=df.shape[0]):
        # User features updates
        features_dicts['cont_preguntas_corr_user'][row.user_id] += row.answered_correctly
        if row.answered_correctly == 0:
            features_dicts['timestamp_u_incorrect'][row.user_id] = row.timestamp
        # ------------------------------------------------------------------
        # Question features updates
        features_dicts['cont_preguntas_corr'][row.content_id] += row.answered_correctly
        
def load_obj(name):
    with open('../input/lgbm-model-riiid-training/' + name + '.pkl',mode =  'rb') as f:
        return pickle.load(f)
    
def inference(TARGET, FEATURES, model, questions, prior_question_elapsed_time_mean, features_dicts):
    
    # Get api iterator and predictor
    env = riiideducation.make_env()
    iter_test = env.iter_test()
    set_predict = env.predict
    
    previous_test_df = None
    for (test_df, sample_prediction_df) in iter_test:
        if previous_test_df is not None:
            previous_test_df[TARGET] = eval(test_df["prior_group_answers_correct"].iloc[0])
            actualizar_features(previous_test_df, features_dicts)
            
            #####
            prev_test_df['answered_correctly'] = eval(test_df['prior_group_answers_correct'].iloc[0])
            prev_test_df = prev_test_df[prev_test_df.content_type_id == False]
            
            prev_group = prev_test_df[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(lambda r: (
                r['content_id'].values,
                r['answered_correctly'].values))
            for prev_user_id in prev_group.index:
                prev_group_content = prev_group[prev_user_id][0]
                prev_group_ac = prev_group[prev_user_id][1]
                if prev_user_id in group.index:
                    group[prev_user_id] = (np.append(group[prev_user_id][0],prev_group_content), 
                                           np.append(group[prev_user_id][1],prev_group_ac))
                else:
                    group[prev_user_id] = (prev_group_content,prev_group_ac)
                if len(group[prev_user_id][0])>MAX_SEQ:
                    new_group_content = group[prev_user_id][0][-MAX_SEQ:]
                    new_group_ac = group[prev_user_id][1][-MAX_SEQ:]
                    group[prev_user_id] = (new_group_content,new_group_ac)
            #####
            
        #####
        prev_test_df = test_df.copy()
        #####
        
        test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
        test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
        
        previous_test_df = test_df.copy()
        
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop = True)
        test_df = test_df.merge(questions, on = 'content_id', how='left')
        test_df[TARGET] = 0
        test_df = añadir_features(test_df, features_dicts)
        
        #####
        test_dataset = TestDataset(group, test_df, skills)
        test_dataloader = DataLoader(test_dataset, batch_size=51200, shuffle=False)

        SAKT_outs = []

        for item in test_dataloader:
            x = item[0].to(device).long()
            target_id = item[1].to(device).long()

            with torch.no_grad():
                output, att_weight = SAKT_model(x, target_id)

            output = torch.sigmoid(output)
            output = output[:, -1]
            SAKT_outs.extend(output.view(-1).data.cpu().numpy())
        #####
        
        test_df[TARGET] = np.array(SAKT_outs) * 0.5 + model.predict(test_df[FEATURES]) * 0.5 #media aritmetica (0.783)
        #####
        
        #test_df[TARGET] =  model.predict(test_df[FEATURES])
        set_predict(test_df[['row_id', TARGET]])
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAKT_model = SAKTModel(n_skill, embed_dim=128)
    try:
        SAKT_model.load_state_dict(torch.load('../input/riiid-sakt-model-training-with-gpu-and-inference/SAKT_model.pt'))
    except:
        SAKT_model.load_state_dict(torch.load('../input/riiid-sakt-model-training-with-gpu-and-inference/SAKT_model.pt', map_location='cpu'))
    
    SAKT_model.to(device)
    SAKT_model.eval()
    
    group = pickle.load(open("group.pkl", "rb"))
    
    print(psutil.virtual_memory().percent)
    
    #ACTUALIZACION DE FEATURES
    columnas = ['timestamp', 'user_id', 'answered_correctly', 'content_id', 'content_type_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']
    dtypes={'user_id': 'int32', 
            'content_id': 'int16',
            'task_container_id': 'int16',
            'content_type_id': 'int8',
            'answered_correctly': 'int8', 
            'prior_question_elapsed_time': 'float32',
            'prior_question_had_explanation': 'boolean',
            'timestamp':'int64',}
    
    
    train = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', usecols = columnas, dtype = dtypes)#, nrows = 1000)
    #train.sort_values(by=['user_id'], inplace = True)
    
    #Creamos dicionarios
    features_dicts = {'cont_preguntas_user': defaultdict(int),
                      'cont_preguntas_corr_user': defaultdict(int),
                      'elapsed_time_u_sum': defaultdict(int),
                      'explanation_u_sum': defaultdict(int),
                      'cont_preguntas': defaultdict(int),
                      'cont_preguntas_corr': defaultdict(int),
                      'elapsed_time_q_sum': defaultdict(int),
                      'explanation_q_sum': defaultdict(int),
                      'intentos_dict': defaultdict(lambda: defaultdict(int)),
                      'timestamp_u': defaultdict(list),
                      'timestamp_u_incorrect': defaultdict(int),
                      'CUMULATIVE_ELO_USER': defaultdict(int)}
    
    #eliminacion de lectures
    train = train.loc[train.content_type_id == False].reset_index(drop = True)
    
    #Limpieza
    prior_question_elapsed_time_mean = train['prior_question_elapsed_time'].dropna().values.mean()
    train['prior_question_had_explanation'] = train.prior_question_had_explanation.fillna(False).astype('int8')
    
    train['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
    
    #QUESTIONS MEAN
    columnas = ['content_id','mean_question_accuracy','std_accuracy']
    dtypes={'content_id': 'int16', 'mean_question_accuracy': 'float32', 'std_accuracy': 'float32'}
    questions_1 = pd.read_csv('../input/question-csv-riiid/question_metadata.csv', usecols = columnas, dtype = dtypes)
    
    train = train.merge(questions_1, on = 'content_id', how='left')
    
    print('ACTUALIZACION DE FEATURES INICIADO')
    actualizar_features_inicio(train, features_dicts)
    print('ACTUALIZACION DE FEATURES FINALIZADO')
    
    del train
    gc.collect()
    
    #Carga de questions
    columnas = ['question_id', 'bundle_id']
    dtypes={'question_id': 'uint16', 'bundle_id': 'uint8'}
    questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv', usecols = columnas, dtype = dtypes)
    
    questions = questions.merge(questions_1, left_on = 'question_id', right_on = 'content_id', how = 'left')
    questions.drop(columns=['question_id'],  inplace = True)
    
    del questions_1, columnas, dtypes
    gc.collect()
    
    model = lgb.Booster(model_file = '../input/lgbm-model-riiid-training/model_1.txt')
    TARGET = load_obj('TARGET')
    FEATURES = load_obj('FEATURES')
    prior_question_elapsed_time_mean = load_obj('prior_question_elapsed_time_mean')
    
    inference(TARGET, FEATURES, model, questions, prior_question_elapsed_time_mean, features_dicts)
