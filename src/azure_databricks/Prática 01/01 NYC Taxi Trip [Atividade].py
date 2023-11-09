# Databricks notebook source
# MAGIC %fs ls dbfs:/databricks-datasets/nyctaxi/tripdata/yellow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leitura dos dados necessários

# COMMAND ----------

df = spark.read.csv("dbfs:/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-12.csv.gz", inferSchema=True, header=True)

# COMMAND ----------

type(df)

# COMMAND ----------

df.select("fare_amount")

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Para facilitar a execução das atividades, quem tiver maior familiaridade com SQL poderá utilizar o comando abaixo que nos permite executar queries SQL diretamente

# COMMAND ----------

df.createOrReplaceTempView("taxi_trips")

# COMMAND ----------

query = """
    SELECT *
    FROM taxi_trips
"""

spark.sql(query).display()

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * from taxi_trips

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perguntas de negócio

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 1: Quais são os horários de maior demanda por táxis em Nova York?

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   CAST(DATE_FORMAT(tpep_pickup_datetime, 'H') AS INT) AS pickup_time,
# MAGIC   COUNT(*) AS num_trips
# MAGIC   FROM taxi_trips
# MAGIC   GROUP BY
# MAGIC   pickup_time
# MAGIC   ORDER BY
# MAGIC   num_trips DESC
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 2: Qual é a média de distância percorrida por viagem em cada dia da semana?

# COMMAND ----------

- dias da semana, preciso extrair
- media por dia da semana

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   tpep_pickup_datetime,
# MAGIC   trip_distance,
# MAGIC   dayofweek(tpep_pickup_datetime) as day_of_week
# MAGIC from taxi_trips

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   dayofweek(tpep_pickup_datetime) as day_of_week,
# MAGIC   avg(trip_distance)
# MAGIC from taxi_trips
# MAGIC group by
# MAGIC   day_of_week

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   dayofweek(tpep_pickup_datetime) as day_of_week,
# MAGIC   count(1)
# MAGIC from taxi_trips
# MAGIC group by
# MAGIC   day_of_week

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 3: Quantas corridas foram pagas em dinheiro por cada dia da semana?

# COMMAND ----------

# MAGIC %sql
# MAGIC     SELECT count(VendorID), date_format (tpep_dropoff_datetime, 'E') a
# MAGIC     FROM taxi_trips
# MAGIC     WHERE payment_type == 2
# MAGIC     group by a
# MAGIC     order by count(VendorID)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 4: Qual é a proporção de viagens curtas (< 2 milhas) para viagens longas (>= 2 milhas) durante os dias úteis?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     DAYOFWEEK(tpep_pickup_datetime) AS diaSemana,
# MAGIC     SUM(CASE WHEN trip_distance < 2 THEN 1 ELSE 0 END) AS viagemCurta,
# MAGIC     SUM(CASE WHEN trip_distance >= 2 THEN 1 ELSE 0 END) AS viagemLonga,
# MAGIC     COUNT(*) AS totalViagem
# MAGIC FROM
# MAGIC     taxi_trips
# MAGIC WHERE
# MAGIC     DAYOFWEEK(tpep_pickup_datetime) BETWEEN 2 AND 6
# MAGIC GROUP BY
# MAGIC     diaSemana
# MAGIC ORDER BY
# MAGIC     diaSemana;

# COMMAND ----------

import pyspark.sql.functions as F

# COMMAND ----------

display(
    df
    .withColumn('trip_type', F.when(
        F.col("trip_distance")<2, "curta"
        ).otherwise("longa")
    )
    .filter(
        F.dayofweek("tpep_pickup_datetime").isin([1,7])
    )
    .groupby("trip_type")
    .count()
)

# COMMAND ----------

display(
    df
    .withColumn('trip_type', F.when(
        F.col("trip_distance")<2, "curta"
        ).otherwise("longa")
    )
    .filter(
        F.dayofweek("tpep_pickup_datetime").between(2,6)
    )
    .groupby("trip_type")
    .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 5: Em quais horários os passageiros dão as melhores gorjetas?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   CAST(DATE_FORMAT(tpep_pickup_datetime, 'H') AS INT) AS hora,
# MAGIC   AVG(tip_amount) AS melhor_gorjeta
# MAGIC FROM
# MAGIC   taxi_trips
# MAGIC GROUP BY
# MAGIC   hora
# MAGIC ORDER BY
# MAGIC   melhor_gorjeta DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   CAST(DATE_FORMAT(tpep_dropoff_datetime, 'H') AS INT) AS hora,
# MAGIC   AVG(tip_amount) AS melhor_gorjeta
# MAGIC FROM
# MAGIC   taxi_trips
# MAGIC GROUP BY
# MAGIC   hora
# MAGIC ORDER BY
# MAGIC   melhor_gorjeta DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta: Quais são os cinco principais locais de partida (PULocationID) que resultam na maior receita total?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 6: Qual é o valor médio das corridas por tipo de pagamento?

# COMMAND ----------

display(
    df
    .groupBy("payment_type")
    .count()
)

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
