# Databricks notebook source
# MAGIC %fs ls dbfs:/databricks-datasets/nyctaxi/tripdata/yellow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leitura dos dados necessários

# COMMAND ----------

df = spark.read.csv("dbfs:/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-12.csv.gz", inferSchema=True, header=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Para facilitar a execução das atividades, quem tiver maior familiaridade com SQL poderá utilizar o comando abaixo que nos permite executar queries SQL diretamente

# COMMAND ----------

df.createOrReplaceTempView("taxi_trips")

query = """
    SELECT *
    FROM taxi_trips
"""

spark.sql(query).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perguntas de negócio

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 1: Quais são os horários de maior demanda por táxis em Nova York?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 2: Qual é a média de distância percorrida por viagem em cada dia da semana?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 3: Quantas corridas foram pagas em dinheiro por cada dia da semana?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 4: Qual é a proporção de viagens curtas (< 2 milhas) para viagens longas (>= 2 milhas) durante os dias úteis?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 5: Em quais horários os passageiros dão as melhores gorjetas?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta: Quais são os cinco principais locais de partida (PULocationID) que resultam na maior receita total?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 6: Qual é o valor médio das corridas por tipo de pagamento?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 7: Como a distância média das viagens varia ao longo do dia?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 8: Qual é a duração média das corridas que começam e terminam na mesma localização?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 9. Qual a proporção de viagens com apenas um passageiro em relação ao total de viagens?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 10. Quais são os top 3 horários de pico para início das corridas?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 11. Qual é a variação percentual na quantidade de viagens entre dias úteis e fins de semana?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 12. Quais são os 5 destinos mais comuns nas corridas que excedem $50?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 13. Como a distância média e o valor médio da gorjeta estão relacionados?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 14. Qual é o tempo médio de viagem entre as localizações mais populares de embarque e desembarque?
