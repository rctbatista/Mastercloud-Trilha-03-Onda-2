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
# MAGIC #### Pergunta 6: Qual é o valor médio das corridas por tipo de pagamento?

# COMMAND ----------

display(
    df
    .groupBy("payment_type")
    .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta Bonus: Quais são os cinco principais locais de partida (PULocationID) que resultam na maior receita total?

# COMMAND ----------

display(df)

# COMMAND ----------

import pyspark.sql.functions as F

display(
    df
    .groupBy(F.col("PULocationID").alias('Localizacao'))
    .agg(
        F.round(F.sum('total_amount'), 2).alias("maior_receita"),
        F.round(F.avg('total_amount'), 2).alias("valor_receita_media")
    )
    .orderBy(
        F.col("maior_receita").desc()
    )
    .limit(5)
)

# COMMAND ----------

display(
    df.summary()
)

# COMMAND ----------

# MAGIC %sql
# MAGIC  
# MAGIC select round(sum(total_amount), 2) as maior_receita
# MAGIC       ,PULocationID
# MAGIC from taxi_trips
# MAGIC group by PULocationID
# MAGIC order by maior_receita desc
# MAGIC limit 5

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 7: Como a distância média das viagens varia ao longo do dia?

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CAST(EXTRACT(HOUR FROM tpep_pickup_datetime) AS INT) AS HoraDoDia, AVG(trip_distance) AS DistanciaMedia
# MAGIC
# MAGIC FROM taxi_trips
# MAGIC
# MAGIC GROUP BY EXTRACT(HOUR FROM tpep_pickup_datetime)
# MAGIC
# MAGIC ORDER BY HoraDoDia ASC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CAST(EXTRACT(HOUR FROM tpep_pickup_datetime) AS INT) AS HoraDoDia, 
# MAGIC AVG(datediff(minute, tpep_pickup_datetime, tpep_dropoff_datetime)) AS tempo_duracao
# MAGIC
# MAGIC FROM taxi_trips
# MAGIC
# MAGIC GROUP BY EXTRACT(HOUR FROM tpep_pickup_datetime)
# MAGIC
# MAGIC ORDER BY HoraDoDia ASC;

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 8: Qual é a duração média das corridas que começam e terminam na mesma localização?

# COMMAND ----------

# MAGIC %sql
# MAGIC  
# MAGIC select count(PULocationID)
# MAGIC from taxi_trips
# MAGIC WHERE PULocationID = DOLocationID

# COMMAND ----------

tempo_duracao = (F.unix_timestamp(F.col("tpep_dropoff_datetime")) - F.unix_timestamp(F.col("tpep_pickup_datetime")))/60
filtro_tempo_duracao = tempo_duracao.between(5, 60)

# COMMAND ----------

def diff_tempos(tempo_final, tempo_inicial):
    col_final = F.unix_timestamp(F.col(tempo_final))
    col_inicial = F.unix_timestamp(F.col(tempo_inicial))

    return (col_final - col_inicial)/60

# COMMAND ----------

diff_tempos('tpep_dropoff_datetime', 'tpep_pickup_datetime')

# COMMAND ----------

tempo_duracao

# COMMAND ----------

display(
    df
    .filter(
        (F.col("PULocationID") == F.col("DOLocationID")) & 
        filtro_tempo_duracao
    )
    .select("PULocationID", tempo_duracao)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC  
# MAGIC SELECT
# MAGIC     datediff(minute, tpep_pickup_datetime, tpep_dropoff_datetime) AS tempo_duracao,
# MAGIC     PULocationID,
# MAGIC     payment_type
# MAGIC FROM
# MAGIC     taxi_trips
# MAGIC WHERE 
# MAGIC   PULocationID = DOLocationID and
# MAGIC   datediff(minute, tpep_pickup_datetime, tpep_dropoff_datetime) BETWEEN 5 AND 60

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 9. Qual a proporção de viagens com apenas um passageiro em relação ao total de viagens?

# COMMAND ----------

from pyspark.sql import functions as F

df.createOrReplaceTempView("taxi_trips")
df_silver = df.filter(df.passenger_count.isNotNull())
total_trips = df_silver.count()

df_ouro = (
    df_silver
    .groupBy(F.col("passenger_count").alias("Número de Passageiros"))
    .agg(
        F.count("passenger_count").alias("Total de Viagens"),
        (F.count("passenger_count") * 100 / total_trips).alias("Percentual de Representatividade")
    )
    .orderBy(
        F.col("passenger_count").asc()
    )
)

df_ouro.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 10. Quais são os top 3 horários de pico para início das corridas?

# COMMAND ----------

from pyspark.sql.functions import  date_format

df.createOrReplaceTempView("taxi_times")

top_3_horarios = (
    df
    .withColumn("pickup_time", date_format("tpep_pickup_datetime", "HH"))
    .groupBy("pickup_time")
    .count()
    .orderBy("count", ascending=False)
    .limit(3)
)
top_3_horarios.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 11. Qual é a variação percentual na quantidade de viagens entre dias úteis e fins de semana?

# COMMAND ----------

total_corridas = df.count()

display(
    df
    .withColumn('Final de Semana', F.when(
        F.dayofweek("tpep_pickup_datetime").isin([1,7]), "0"
        ).otherwise("1")
    )
    .groupby(
        F.col('Final de Semana')
    )
    .agg(F.count('Final de Semana')/total_corridas)
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Pergunta 12. Quais são os 5 destinos mais comuns nas corridas que excedem $50?

# COMMAND ----------

from pyspark.sql.functions import col,desc

display(
    df
    .filter(col("fare_amount") >= 50.0)
    .groupBy(col("DOLocationID"))
    .count()
    .sort(desc("count"))
    .limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 13. Como a distância média e o valor médio da gorjeta estão relacionados?

# COMMAND ----------

distancia = (F.when(
        F.col("trip_distance").between(0, 2.0), "dist_curta"
    )
    .when(
        F.col("trip_distance").between(2.01, 10.0), "dist_media"
    )
    .when(
        F.col("trip_distance").between(10.1, 100.), "dist_longa"
    )
    .otherwise("NA")
)

# COMMAND ----------

distancia

# COMMAND ----------

display(
    df
    .groupby(
        distancia
    )
    .count()
)

# COMMAND ----------

display(
    df
    .groupby(
        distancia
    )
    .agg(
        F.avg("fare_amount")
    )
)

# COMMAND ----------

display(
    df
    .groupby(
        distancia
    )
    .agg(
        F.avg("tip_amount")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 14. Qual é o tempo médio de viagem entre as localizações mais populares de embarque e desembarque?

# COMMAND ----------

df

# COMMAND ----------

display(
    df
    .filter(filtro_tempo_duracao)
    .groupby("PULocationID", "DOLocationID", date_format("tpep_pickup_datetime", "HH"))
    .agg(
        F.avg(tempo_duracao).alias("tempo_medio_da_corrida"),
        F.min(tempo_duracao).alias("tempo_min_da_corrida"),
        F.max(tempo_duracao).alias("tempo_max_da_corrida"),
        F.count(F.col("VendorID").alias("qtd_corridas"))
    )
    .orderBy(F.col("PULocationID").desc(), F.col("DOLocationID").asc(), date_format("tpep_pickup_datetime", "HH").asc())
)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import unix_timestamp, avg
from pyspark.sql.window import Window

# PySpark API
windowSpec  = Window.partitionBy("PULocationID", "DOLocationID")

df.filter(filtro_tempo_duracao) \
    .withColumn("trip_duration", 
              (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime"))/60) \
  .withColumn("average_trip_duration", avg("trip_duration").over(windowSpec)) \
  .withColumn("diff_trip_duration_and_average", F.col("trip_duration") - F.col("average_trip_duration")) \
  .orderBy("diff_trip_duration_and_average", ascending=False) \
  .display()

# COMMAND ----------


