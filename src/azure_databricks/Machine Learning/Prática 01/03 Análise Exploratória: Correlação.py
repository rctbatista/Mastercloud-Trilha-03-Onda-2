# Databricks notebook source
# MAGIC %md
# MAGIC ### Toy Example: Correlação entre Duas Variáveis
# MAGIC
# MAGIC Vamos supor que queremos analisar a correlação entre a "Idade" e a "Renda" de um grupo de indivíduos.
# MAGIC
# MAGIC #### Passo 1: Geração de Dados Sintéticos
# MAGIC
# MAGIC Vamos criar um conjunto de dados com 100 observações, onde:
# MAGIC
# MAGIC     Idade: Valores entre 20 e 60 anos.
# MAGIC     Renda: Uma função da idade, adicionando algum ruído aleatório para simular variações reais.

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)  # Para resultados reproduzíveis

# Gerar dados
n = 100
idades = np.random.randint(20, 60, size=n)
rendas = idades * 20 + np.random.normal(0, 100, n)  # Renda como função da idade com ruído

# Criar DataFrame
df = pd.DataFrame({'Idade': idades, 'Renda': rendas})

# Visualizar os primeiros dados
print(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Passo 2: Visualização com Scatter Plot
# MAGIC
# MAGIC Um scatter plot ajuda a visualizar a relação entre as duas variáveis.

# COMMAND ----------

plt.scatter(df['Idade'], df['Renda'])
plt.title('Relação entre Idade e Renda')
plt.xlabel('Idade')
plt.ylabel('Renda')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Passo 3: Cálculo da Correlação
# MAGIC
# MAGIC Vamos calcular a correlação de Pearson.

# COMMAND ----------

correlacao = df.corr()
print("Matriz de Correlação:\n", correlacao)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Passo 4: Interpretação
# MAGIC
# MAGIC  - Um coeficiente de correlação próximo de +1 indica uma forte correlação positiva (à medida que uma variável aumenta, a outra também aumenta).
# MAGIC  - Um valor próximo de -1 indica uma forte correlação negativa (à medida que uma variável aumenta, a outra diminui).
# MAGIC  Valores próximos de 0 indicam uma falta de correlação linear entre as variáveis.

# COMMAND ----------

sns.heatmap(correlacao, annot=True, cmap='coolwarm')
plt.title('Heatmap de Correlação entre Idade e Renda')
plt.show()
