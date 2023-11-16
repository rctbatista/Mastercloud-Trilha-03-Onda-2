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
rendas = idades * 100 + np.random.normal(0, 1000, n)  # Renda como função da idade com ruído

# Criar DataFrame
df = pd.DataFrame({'Idade': idades, 'Renda': rendas})

# COMMAND ----------

# Visualizar os primeiros dados
display(df)

# COMMAND ----------

df['Idade'].describe(), df['Renda'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Passo 2: Visualização com Scatter Plot
# MAGIC
# MAGIC Um scatter plot ajuda a visualizar a relação entre as duas variáveis.

# COMMAND ----------

df['Idade'].hist()

# COMMAND ----------

df['Renda'].hist()

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

# Como visualizar a tendência?
sns.lmplot(x='Idade', y='Renda', data=df, ci=None)
plt.title('Relação Linear entre Idade e Renda com Linha de Tendência')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Passo 4: Interpretação
# MAGIC
# MAGIC  - Um coeficiente de correlação próximo de +1 indica uma forte correlação positiva (à medida que uma variável aumenta, a outra também aumenta).
# MAGIC  - Um valor próximo de -1 indica uma forte correlação negativa (à medida que uma variável aumenta, a outra diminui).
# MAGIC  Valores próximos de 0 indicam uma falta de correlação linear entre as variáveis.

# COMMAND ----------

sns.heatmap(correlacao, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Heatmap de Correlação entre Idade e Renda')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Um exemplo onde podemos explorar outros comportamentos entre variáveis

# COMMAND ----------

# Relação senoidal entre idade e salário
salarios_sen = 5000 + 3000 * np.sin(0.18 * idades) + np.random.normal(0, 100, n)

# DataFrame
df_sen = pd.DataFrame({'Idade': idades, 'Salario': salarios_sen})

# Calculando a correlação
print("Correlação (relação senoidal):", df_sen.corr().iloc[0, 1])

# Calculando a correlação não linear
print("Correlação Spearman (relação senoidal):", df_sen.corr('spearman').iloc[0, 1])


# Visualização
sns.scatterplot(x='Idade', y='Salario', data=df_sen)
plt.title('Relação Senoidal entre Idade e Salário')
plt.show()

# COMMAND ----------

# Exemplo com a relação quadrática
coeficientes = np.polyfit(df_sen['Idade'], df_sen['Salario'], 4)
polinomio = np.poly1d(coeficientes)
x = np.linspace(min(df_sen['Idade']), max(df_sen['Idade']), 100)

plt.scatter(df_sen['Idade'], df_sen['Salario'])
plt.plot(x, polinomio(x), color='red')  # Linha de tendência
plt.title('Relação Quadrática entre Idade e Salário com Linha de Tendência')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.show()


# COMMAND ----------


