# Databricks notebook source
import  pyspark.sql.functions as F
import pandas as pd

# COMMAND ----------

repository_path = "file:/Workspace/Repos/emanuelfontelles@hotmail.com/Mastercloud-Trilha-03-Onda-2/"
data_path = repository_path + "src/azure_databricks/data/"
file_path = data_path + "input/treino.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Leitura dos dados que foram previamente extraidos do banco de dados/datalake

# COMMAND ----------

df = spark.read.csv(file_path, header=True, inferSchema=True)

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movendo dados para camada Bronze

# COMMAND ----------

df.write.parquet(data_path+"bronze/credito_inadimplencia", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movendo dados para camada Silver

# COMMAND ----------

bronze_df = spark.read.parquet(data_path+"bronze/credito_inadimplencia")

# COMMAND ----------

bronze_df = bronze_df.withColumn('idade', F.col('idade').cast('int'))
bronze_df = bronze_df.withColumn('salario_mensal', F.col('salario_mensal').cast('float'))
bronze_df = bronze_df.withColumn('razao_debito', F.col('razao_debito').cast('float'))
bronze_df = bronze_df.withColumn('inadimplente', F.col('inadimplente').cast('int'))
bronze_df = bronze_df.withColumn('numero_emprestimos_imobiliarios', F.col('numero_emprestimos_imobiliarios').cast('float'))
bronze_df = bronze_df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))
bronze_df = bronze_df.withColumn('vezes_passou_de_30_59_dias', F.col('vezes_passou_de_30_59_dias').cast('float'))
bronze_df = bronze_df.withColumn('numero_de_vezes_que_passou_60_89_dias', F.col('numero_de_vezes_que_passou_60_89_dias').cast('float'))
bronze_df = bronze_df.withColumn('numero_vezes_passou_90_dias', F.col('numero_vezes_passou_90_dias').cast('float'))
bronze_df = bronze_df.withColumn('numero_linhas_crdto_aberto', F.col('numero_linhas_crdto_aberto').cast('int'))
bronze_df = bronze_df.withColumn('util_linhas_inseguras', F.col('util_linhas_inseguras').cast('float'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### O que deveria ser feito na etapa Bronze-Silver:
# MAGIC   - Limpeza e correção de tipos
# MAGIC   - Enriquecimento com possíveis novas variáveis
# MAGIC   - Remoção de valores faltantes
# MAGIC   - Analise exploratória

# COMMAND ----------

bronze_df.write.parquet(data_path+"silver/credito_inadimplencia", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Movendo dados para camada Gold

# COMMAND ----------

silver_df = spark.read.parquet(data_path+"silver/credito_inadimplencia")

# COMMAND ----------

silver_df.write.parquet(data_path+"gold/credito_inadimplencia", mode="overwrite")
