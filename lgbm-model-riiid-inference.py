%reset -f

import riiideducation
import pandas as pd
import numpy as np
import gc
from collections import defaultdict
from tqdm.notebook import tqdm
import pickle
import lightgbm as lgb


# %load_ext line_profiler

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
    
    
    del tasa_acierto_usuario, cont_preguntas_user_f,CUMULATIVE_ELO_USER, cont_preguntas_corr_user_f, elapsed_time_u_avg, explanation_u_avg, tasa_acierto_pregunta, elapsed_time_q_avg, explanation_q_avg, intentos, user_pause_timestamp_1,  user_pause_timestamp_2,user_pause_timestamp_3, user_pause_timestamp_incorrect
 
    df = pd.concat([df, user_df], axis = 1)
    del user_df
    
    #Features extra
    df['correction'] = (df['user_pause_timestamp_1'] + 1) / (df['user_pause_timestamp_incorrect'] + 1) + df['prior_question_had_explanation'] + df['intentos']
    df['user_pause_timestamp_ratio_1'] = (df['user_pause_timestamp_1'] + 1)/ (df['user_pause_timestamp_2'] + 1)
    df['%_media_armonica'] = 2*df['%_acierto_usuario']*df['mean_question_accuracy']/(df['%_acierto_usuario'] + df['mean_question_accuracy'])
    df['%_media_armonica'].fillna(0.642673913, inplace = True)

    df['user_pause_timestamp_MEAN_RATIO'] = df['user_pause_timestamp_1']/((df['user_pause_timestamp_1'] + df['user_pause_timestamp_2'] + df['user_pause_timestamp_3'])/3 + 1)
    df['ELO'] = (df['CUMULATIVE_ELO_USER'] + 4*df['user_pause_timestamp_MEAN_RATIO']*(2*df['cont_preguntas_corr_user'] - df['cont_preguntas_user'])) / df['cont_preguntas_user']
    df['expected_prob'] = 1 / (1 + 9**(((1 - df['mean_question_accuracy']) * 100 - df['ELO'])/15))

    df.replace(np.inf, 0, inplace = True)
    df[['ELO','correction','user_pause_timestamp_ratio_1','%_media_armonica','user_pause_timestamp_MEAN_RATIO','expected_prob']] = df[['ELO','correction','user_pause_timestamp_ratio_1','%_media_armonica','user_pause_timestamp_MEAN_RATIO','expected_prob']].astype(np.float16)
    
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
                  'CUMULATIVE_ELO_USER': defaultdict(int),
                  'True_ELO':defaultdict(int)}

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

        test_df['prior_question_had_explanation'] = test_df.prior_question_had_explanation.fillna(False).astype('int8')
        test_df['prior_question_elapsed_time'].fillna(prior_question_elapsed_time_mean, inplace = True)
        
        previous_test_df = test_df.copy()
        
        test_df = test_df[test_df['content_type_id'] == 0].reset_index(drop = True)
        test_df = test_df.merge(questions, on = 'content_id', how='left')
        test_df[TARGET] = 0
        test_df = añadir_features(test_df, features_dicts)
        test_df[TARGET] =  model.predict(test_df[FEATURES])
        set_predict(test_df[['row_id', TARGET]])
        
        
if __name__ == "__main__":
    #Carga de questions
    columnas = ['question_id','part', 'bundle_id']
    dtypes={'question_id': 'int16', 'part': 'uint8', 'bundle_id': 'uint8'}
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