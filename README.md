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
