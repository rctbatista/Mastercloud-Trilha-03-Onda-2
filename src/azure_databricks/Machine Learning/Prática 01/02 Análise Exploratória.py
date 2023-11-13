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
# MAGIC ### Leitura dos dados que foram previamente extraidos do banco de dados/datalake

# COMMAND ----------

df_spark = spark.read.parquet(data_path + "silver/credito_inadimplencia")
df = df_spark.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Analise do conteúdo do arquivo

# COMMAND ----------

# Visualizar as primeiras linhas
display(df)

# COMMAND ----------

# Informações gerais sobre o dataset
print(df.info())

# COMMAND ----------

# Estatísticas descritivas
print(df.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Análise exploratória dos dados (AED)
# MAGIC AED é o processo de análise inicial dos dados para descobrir padrões, identificar anomalias, testar hipóteses e verificar suposições com a ajuda de estatísticas resumidas e representações gráficas.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Histograma
# MAGIC Conceito:
# MAGIC  - Histogramas são usados para representar a distribuição de uma variável contínua, mostrando a frequência de diferentes intervalos de valores.
# MAGIC  - Útil para identificar distribuições (normal, assimétrica, etc.), presença de outliers e compreender a variação dos dados.

# COMMAND ----------

# Histograma para uma variável
df['idade'].hist(bins=20)
plt.title('Distribuição de Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# COMMAND ----------

df["salario_mensal"].hist(bins=100, range=(0, 50e3))
plt.title('Distribuição do salário')
plt.xlabel('Salário')
plt.ylabel('Frequência')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

df.hist(bins=50, figsize=(20,15))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Boxplots
# MAGIC Conceito:
# MAGIC  - Boxplots são úteis para visualizar a distribuição de dados e identificar outliers.
# MAGIC  - Eles mostram a mediana, quartis e valores extremos.

# COMMAND ----------

sns.boxplot(x='inadimplente', y='idade', data=df)
plt.title('Boxplot de Idade por Inadimplência')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scatter Plots (Nuvem de pontos)
# MAGIC Conceito:
# MAGIC  - Scatter plots são utilizados para visualizar a relação entre duas variáveis quantitativas.
# MAGIC  - Úteis para identificar padrões de correlação.

# COMMAND ----------

plt.scatter(df['idade'], df['numero_emprestimos_imobiliarios'], alpha=0.5)
plt.title('Scatter Plot de Idade vs Quantidade de Emprestimos Imobilários')
plt.xlabel('Idade')
plt.ylabel('Emprestimos Imobilários')
plt.show()

# COMMAND ----------

plt.scatter(df['idade'], df['salario_mensal'])
plt.title('Scatter Plot de Idade vs Renda')
plt.xlabel('Idade')
plt.ylabel('Renda')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlação
# MAGIC Conceito:
# MAGIC  - A correlação mede a relação estatística entre duas variáveis.
# MAGIC  - O coeficiente de correlação varia entre -1 e 1. Valores próximos a 1 indicam forte correlação positiva, enquanto valores próximos a -1 indicam forte correlação negativa.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construção de novas variáveis pertinentes para nosso problema

# COMMAND ----------

# Criação de novas variáveis (feature engineering)

# 2. Interação entre idade e número de linhas de crédito
df['idade_x_linhas_credito'] = df['idade'] * df['numero_linhas_crdto_aberto']

# 3. Razão entre salário mensal e dívida
df['razao_salario_debito'] = df['salario_mensal'] / df['razao_debito']
df['razao_salario_debito'] = df['razao_salario_debito'].replace([np.inf, -np.inf], np.nan)  # Tratamento para divisão por zero

# 4. Total de atrasos
df['total_atrasos'] = df['vezes_passou_de_30_59_dias'] + df['numero_vezes_passou_90_dias'] + df['numero_de_vezes_que_passou_60_89_dias']

# 5. Indicador de dados faltantes
df['salario_mensal_faltante'] = df['salario_mensal'].isnull().astype(int)

# Exibir as primeiras linhas do DataFrame com as novas variáveis
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Após a construção de novas informações salvamos o novo dataset na nossa camada `gold`

# COMMAND ----------

df_silver = spark.createDataFrame(df)

# COMMAND ----------

df_silver.write.parquet(data_path+"gold/credito_inadimplencia", mode="overwrite")
