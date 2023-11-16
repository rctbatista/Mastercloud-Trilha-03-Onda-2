# Databricks notebook source
# MAGIC %sh apt install graphviz -y

# COMMAND ----------

!pip install graphviz
dbutils.library.restartPython()

# COMMAND ----------

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

display(df)

# COMMAND ----------

df['total_atrasos'].value_counts(normalize=True).sort_index().plot.bar()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Separação do conjunto de treinamento e teste

# COMMAND ----------

y = df["inadimplente"]
X = df.drop(columns="inadimplente")

# COMMAND ----------

X.shape

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

display(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Utilizaremos um modelo do tipo Boosting para treinamento dos nossos dados

# COMMAND ----------

from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# COMMAND ----------

LGBMClassifier

# COMMAND ----------

# Criação e treinamento do modelo com LightGBM
model = LGBMClassifier(objective='binary', class_weight='balanced')
model.fit(X_train, y_train)

# COMMAND ----------

display(X_test)

# COMMAND ----------

X_test.iloc[0]

# COMMAND ----------

model.predict(X_test)[0]

# COMMAND ----------

model.predict_proba(X_test)[1]

# COMMAND ----------

1000*model.predict_proba(X_test)[1,0]

# COMMAND ----------

y_test.iloc[1]

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

# MAGIC %md
# MAGIC #### Como entender o que o modelo aprendeu?

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Visualizando a Árvore de Decisão
# MAGIC Você pode visualizar árvores individuais no modelo LightGBM para entender as decisões tomadas em cada nó. Isso pode ser feito com a função plot_tree do LightGBM, mas, como mencionado, essa visualização pode ser complexa para modelos com muitas árvores ou árvores profundas.

# COMMAND ----------

import lightgbm as lgb

# Plotando a primeira árvore de decisão
lgb.plot_tree(model, figsize=(40, 20), show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Importância das variáveis
# MAGIC Um modo mais direto de entender o que o modelo está aprendendo é através da importância das características (features). Isso não mostra as regras exatas, mas indica quais características são mais importantes para as decisões do modelo.

# COMMAND ----------

# Obtendo a importância das características
importance = model.feature_importances_

# Plotando a importância
lgb.plot_importance(model, importance_type='split')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Shap Values
# MAGIC SHAP (SHapley Additive exPlanations) é uma abordagem avançada que explica a saída do modelo LightGBM. Ela atribui a cada característica um valor que indica o quanto essa característica contribuiu, positiva ou negativamente, para cada previsão.

# COMMAND ----------

import shap

# Calculando SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Visualização de SHAP values
shap.summary_plot(shap_values, X)

# COMMAND ----------

# Criando o objeto SHAP explainer
explainer = shap.Explainer(model)

# Calculando SHAP values
shap_values = explainer(X)

# Para classificação binária, escolha a classe positiva (índice 1)
shap_values_class = shap_values[...,1]

# Visualização de SHAP values com beeswarm plot
shap.plots.beeswarm(shap_values_class)

# COMMAND ----------



# COMMAND ----------

# End the MLflow run
mlflow.end_run()

# COMMAND ----------

import time
time.sleep(3600*2)
