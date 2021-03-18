# Riiid-Answer-Correctness-Prediction
Personalización de la experiencia de aprendizaje de los estudiantes mediante el desarrollo de un modelo que predice la probabilidad de acertar en la siguiente pregunta en base al desempeño pasado del estudiante.

Esta es la solución que me ha permitido clasificarme en el top 9% mundial en la competición de Kaggle: [Riiid! Answer Correctness Prediction](https://www.kaggle.com/c/riiid-test-answer-prediction/overview).

La solución aportada está compuesta por la mezla de dos modelos.

- El primero de ellos es un modelo basado en LGBM que requiere un 'feature engineering' previo con el fin de maximizar su potencial predictivo. Este modelo, de forma indiviual, obtiene un AUC de 0.787. Este modelo, se acerca bastante al maximo existente en el estado del arte de ese momento (0.791 AUC)
- El segundo modelo parte de la premisa de que las respuestas aportadas por el estudiante, es una sequencia de respuestas de una sequencia de actividades de aprendizaje. Este modelo esta basado en los transformers
