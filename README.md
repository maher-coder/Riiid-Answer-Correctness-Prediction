# Riiid-Answer-Correctness-Prediction
Personalización de la experiencia de aprendizaje de los estudiantes mediante el desarrollo de un modelo que predice la probabilidad de acertar en la siguiente pregunta en base al desempeño pasado del estudiante.

Esta es la solución que me ha permitido clasificarme en el top 9% mundial en la competición de Kaggle: [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction/overview).

La solución aportada está compuesta por la mezla de dos modelos.

- El primero de ellos es un modelo basado en LGBM que requiere un 'feature engineering' previo con el fin de maximizar su potencial predictivo. Este modelo, de forma indiviual, obtiene un AUC de 0.787. Este modelo, se acerca bastante al maximo existente en el estado del arte de ese momento (0.791 AUC)
- El segundo modelo parte de la premisa de que las respuestas aportadas por el estudiante, es una sequencia de respuestas de una sequencia de actividades de aprendizaje. Este modelo esta basado en la arquitectura que ha permitido realizar grandes avances en NLP: los transformers


## Feature engineering
Originalmente, los datos proporcionados por Riiid! son los siguientes:

#### train.csv
 - row_id: (int64) ID code for the row.
 - timestamp: (int64) the time in milliseconds between this user interaction and the first event completion from that user.
 - user_id: (int32) ID code for the user.
 - content_id: (int16) ID code for the user interaction
 - content_type_id: (int8) 0 if the event was a question being posed to the user, 1 if the event was the user watching a lecture.
 - task_container_id: (int16) Id code for the batch of questions or lectures. For example, a user might see three questions in a row before seeing the explanations for any of them. Those three would all share a task_container_id.
 - user_answer: (int8) the user's answer to the question, if any. Read -1 as null, for lectures.
 - answered_correctly: (int8) if the user responded correctly. Read -1 as null, for lectures.
 - prior_question_elapsed_time: (float32) The average time in milliseconds it took a user to answer each question in the previous question bundle, ignoring any lectures in between. Is null for a user's first question bundle or lecture. Note that the time is the average time a user took to solve each question in the previous bundle.
 - prior_question_had_explanation: (bool) Whether or not the user saw an explanation and the correct response(s) after answering the previous question bundle, ignoring any lectures in between. The value is shared across a single question bundle, and is null for a user's first question bundle or lecture. Typically the first several questions a user sees were part of an onboarding diagnostic test where they did not get any feedback.

#### questions.csv: metadata for the questions posed to users.
 - question_id: foreign key for the train/test content_id column, when the content type is question (0).
 - bundle_id: code for which questions are served together.
 - correct_answer: the answer to the question. Can be compared with the train user_answer column to check if the user was right.
 - part: the relevant section of the TOEIC test.
 - tags: one or more detailed tag codes for the question. The meaning of the tags will not be provided, but these codes are sufficient for clustering the questions together.

#### lectures.csv: metadata for the lectures watched by users as they progress in their education.
 - lecture_id: foreign key for the train/test content_id column, when the content type is lecture (1).
 - part: top level category code for the lecture.
 - tag: one tag codes for the lecture. The meaning of the tags will not be provided, but these codes are sufficient for clustering the lectures together.
 - type_of: brief description of the core purpose of the lecture

A partir de estos datos, hemos obtenido los siguientes:
 - User_pause_timestamp_1: Tiempo transcurrido desde la última vez que respondió el alumno a una pregunta
 - User_pause_timestamp_2: Tiempo transcurrido desde la penúltima vez que respondió el alumno a una pregunta
 - User_pause_timestamp_3: Tiempo transcurrido desde la antepenúltima vez que respondió el alumno a una pregunta
 - User_pause_timestamp_ratio_1: (user_pause_timestamp_1 + 1)/ (user_pause_timestamp_2 + 1)
 - %_acierto_usuario: acierto total del alumno / preguntas totales del alumno
 - intentos: Cuantas veces ha respondido a la misma pregunta
 - user_pause_timestamp_incorrect: Tiempo transcurrido desde la última vez que fallo una pregunta
 - correction: (user_pause_timestamp_1 + 1) / (user_pause_timestamp_incorrect + 1) + prior_question_had_explanation + intentos
 - mean_question_accuracy: Tasa de acierto media historica de la pregunta en cuestión
 - explanation_q_avg: Porcentaje de alumnos que han visto la explicación de por que han fallado en el la pregunta en cuestión
 - elapsed_time_q_avg: Tiempo medio que pasa tras responder la última pregunta cuando responden la pregunta en cuestión
 - user_pause_timestamp_MEAN_RATIO: user_pause_timestamp_1/((user_pause_timestamp_1 + user_pause_timestamp_2 + user_pause_timestamp_3)/3 + 1)
 - ELO: Sistema de ELO modificado derivado del ajedrez
 - %_media_armónica: 2*%_acierto_usuario']*mean_question_accuracy']/(%_acierto_usuario + mean_question_accuracy)
 - expected_prob: Probabilidad de acierto en base al ELO -> 1 / (1 + 9**(((1 - mean_question_accuracy) * 100 - ELO)/15))
 - elapsed_time_u_avg: Tiempo medio que pasa tras responder la última pregunta de un alumno
 - CUMULATIVE_ELO_USER: Suma acumulada del histórico de ELOs del alumno
 - std_accuracy: Desviación tipica de la tasa de acierto de la pregunta en cuestión
 - cont_preguntas_user: Contador de preguntas respondidas por el alumno
 - %_acierto_pregunta_CONT: Tasa de acierto media instantanea de la pregunta en cuestión

Tras procesar las anteriores características en el modelo LGBM, obtenemos el siguiente ranking de importancia causal de predicción:
![image](https://user-images.githubusercontent.com/47561659/111660190-1ea54780-880e-11eb-8e50-7eb188dbcb11.png)
## Modelo LGBM

Para entrenar este modelo, esta es la configuración paramétrica que mejores resultados me ha dado:
```
params = {'num_leaves': 350,
          'max_bin':700,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.7,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.05,
          'boosting_type': "gbdt",
          'bagging_seed': 11,
          'metric': 'auc',
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47}
```
Tras 350 rondas de entrenamiento, obtenemos:

![image](https://user-images.githubusercontent.com/47561659/111665514-1b608a80-8813-11eb-8769-ae2c740987fb.png)

## Modelo SAKT
El modelo SAKT está basado en el siguiente [paper](https://arxiv.org/pdf/1907.06837.pdf).
Soluciona uno de los principales problemas que surgian al aplicar un modelo secuencial como las RNN en este tipo de problemas: Poder generalizar a partir de una escasez relativa de datos pasados. Estos se consigue gracias a la capa de 'Atención'.

Arquitectura:

![image](https://user-images.githubusercontent.com/47561659/111663617-49dd6600-8811-11eb-85a9-4248490b344d.png)
![image](https://user-images.githubusercontent.com/47561659/111663691-5792eb80-8811-11eb-8fa6-02d645ea7537.png)

# Agradecimientos
A Kaggle y Riiid! por organizar esta desafiante competición
