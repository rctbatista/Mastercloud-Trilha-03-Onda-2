# Databricks notebook source
import  pyspark.sql.functions as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

repository_path = "file:/Workspace/Repos/emanuelfontelles@hotmail.com/Mastercloud-Trilha-03-Onda-2/"
data_path = repository_path + "src/azure_databricks/data/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tomamos os dados limpos já transformado para a etapa de modelagem

# COMMAND ----------

df_spark = spark.read.parquet(data_path + "gold/credito_inadimplencia")
df = df_spark.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Separação do conjunto de treinamento e teste

# COMMAND ----------

y = df["inadimplente"]
X = df.drop(columns="inadimplente")

# COMMAND ----------

# Removemos a variável resposta do nosso conjunto de treinamento
display(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Dividimos o nosso conjunto `df` inicial para que possamos realizar a etapa de treinamento e validação independentemente

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# COMMAND ----------

X_train.shape, X_test.shape

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Utilizaremos um modelo do tipo Boosting para treinamento dos nossos dados

# COMMAND ----------

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# COMMAND ----------

# Criação e treinamento do modelo com LightGBM
model = LGBMClassifier(objective='binary', class_weight='balanced')
model.fit(X_train, y_train)

# COMMAND ----------

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# COMMAND ----------

display(y_pred)

# COMMAND ----------

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(accuracy)
print(classification_rep)

# COMMAND ----------

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Plota a matriz de confusão
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predito pelo Modelo')
plt.ylabel('Inadimplentes reais')
plt.title('Matriz de confusão')
plt.show()

# COMMAND ----------

# Verifique a versão das bibliotecas
import shap
import lightgbm
print("SHAP version:", shap.__version__)
print("LightGBM version:", lightgbm.__version__)

# SHAP Interpretability
try:
    explainer = shap.Explainer(model.booster_)
    shap_values = explainer(X_test)
    
    # Verifique os valores de SHAP
    print("SHAP values:", shap_values)
    
    # Log SHAP values (you can also visualize them)
    shap.summary_plot(shap_values, X_test)
except Exception as e:
    print("An error occurred:", e)

# COMMAND ----------

explainer = shap.Explainer(model.booster_)
shap_values = explainer(X)

# Verifique os valores de SHAP
print("SHAP values:", shap_values)

# Log SHAP values (you can also visualize them)
shap.summary_plot(shap_values, X)

# COMMAND ----------

# COMMAND ----------

# End the MLflow run
mlflow.end_run()

# COMMAND ----------

import time
time.sleep(3600*2)
