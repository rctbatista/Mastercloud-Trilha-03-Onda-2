# Databricks notebook source
# MAGIC %fs ls dbfs:/databricks-datasets/nyctaxi/tripdata/yellow

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks-datasets/nyctaxi/tripdata/green

# COMMAND ----------

df = spark.read.csv("dbfs:/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-12.csv.gz", inferSchema=True, header=True)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, unix_timestamp
df = df.withColumn("trip_duration", 
              (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime"))/60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perguntas de negócio

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 1: Quais são os horários de maior demanda por táxis em Nova York?

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Calculate trips per hour
demand_df = df.withColumn("hour", F.hour("tpep_pickup_datetime")) \
              .groupBy("hour") \
              .agg(F.count("*").alias("number_of_trips")) \
              .orderBy("number_of_trips", ascending=False)

demand_df.show()

# COMMAND ----------

# Register the DataFrame as a SQL temporary view
df.createOrReplaceTempView("taxi_trips")

query = """
SELECT HOUR(tpep_pickup_datetime) AS hour, COUNT(*) AS number_of_trips
FROM taxi_trips
GROUP BY hour
ORDER BY number_of_trips DESC
"""

spark.sql(query).show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 2: Qual é a média de distância percorrida por viagem em cada dia da semana?

# COMMAND ----------

from pyspark.sql.functions import dayofweek, avg

# PySpark API
df.withColumn("day_of_week", dayofweek(df.tpep_pickup_datetime)) \
  .groupBy("day_of_week") \
  .agg(avg("trip_distance").alias("average_distance")) \
  .orderBy("day_of_week").show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT DAYOFWEEK(tpep_pickup_datetime) AS day_of_week, 
         AVG(trip_distance) AS average_distance
  FROM taxi_trips
  GROUP BY day_of_week
  ORDER BY day_of_week
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 3: Quantas corridas foram pagas em dinheiro por cada dia da semana?

# COMMAND ----------

# PySpark API
df.filter(df.payment_type == 2) \
  .withColumn("day_of_week", dayofweek(df.tpep_pickup_datetime)) \
  .groupBy("day_of_week") \
  .count() \
  .orderBy("day_of_week").show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT DAYOFWEEK(tpep_pickup_datetime) AS day_of_week, 
         COUNT(*) as cash_trips_count
  FROM taxi_trips
  WHERE payment_type = 2
  GROUP BY day_of_week
  ORDER BY day_of_week
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 4: Qual é a proporção de viagens curtas (< 2 milhas) para viagens longas (>= 2 milhas) durante os dias úteis?

# COMMAND ----------

from pyspark.sql.functions import when, col

# COMMAND ----------

# PySpark API
df.withColumn("trip_type", when(df.trip_distance < 2, "Short").otherwise("Long")) \
  .filter(dayofweek(df.tpep_pickup_datetime).between(2, 6)) \
  .groupBy("trip_type") \
  .count() \
  .withColumn("proportion", col("count") / sum("count").over(Window.partitionBy())) \
  .show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT trip_type, COUNT(*) as trip_count, 
         COUNT(*) / SUM(COUNT(*)) OVER () AS proportion
  FROM (
    SELECT CASE 
           WHEN trip_distance < 2 THEN 'Short' 
           ELSE 'Long' 
           END AS trip_type
    FROM taxi_trips
    WHERE DAYOFWEEK(tpep_pickup_datetime) BETWEEN 2 AND 6
  ) AS sub
  GROUP BY trip_type
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 5: Em quais horários os passageiros dão as melhores gorjetas?

# COMMAND ----------

from pyspark.sql.functions import hour, avg

# PySpark API
df.withColumn("pickup_hour", hour(df.tpep_pickup_datetime)) \
  .groupBy("pickup_hour") \
  .agg(avg("tip_amount").alias("average_tip")) \
  .orderBy("average_tip", ascending=False).show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT HOUR(tpep_pickup_datetime) AS pickup_hour, 
         AVG(tip_amount) AS average_tip
  FROM taxi_trips
  GROUP BY pickup_hour
  ORDER BY average_tip DESC
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta: Quais são os cinco principais locais de partida (PULocationID) que resultam na maior receita total?

# COMMAND ----------

from pyspark.sql.functions import sum

# PySpark API
df.groupBy("PULocationID") \
  .agg(sum("total_amount").alias("total_revenue")) \
  .orderBy("total_revenue", ascending=False) \
  .limit(5).show()


# COMMAND ----------

# SQL
spark.sql("""
  SELECT PULocationID, 
         SUM(total_amount) AS total_revenue
  FROM taxi_trips
  GROUP BY PULocationID
  ORDER BY total_revenue DESC
  LIMIT 5
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 6: Qual é o valor médio das corridas por tipo de pagamento?

# COMMAND ----------

from pyspark.sql.functions import mean

# PySpark API
df.groupBy("payment_type") \
  .agg(mean("total_amount").alias("average_fare")) \
  .orderBy("payment_type").show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT payment_type, 
         AVG(total_amount) AS average_fare
  FROM taxi_trips
  GROUP BY payment_type
  ORDER BY payment_type
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 7: Como a distância média das viagens varia ao longo do dia?

# COMMAND ----------

# PySpark API
df.withColumn("hour", hour(df.tpep_pickup_datetime)) \
  .groupBy("hour") \
  .agg(avg("trip_distance").alias("average_distance")) \
  .orderBy("hour").show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT HOUR(tpep_pickup_datetime) AS hour, 
         AVG(trip_distance) AS average_distance
  FROM taxi_trips
  GROUP BY hour
  ORDER BY hour
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 8: Qual é a duração média das corridas que começam e terminam na mesma localização?

# COMMAND ----------

from pyspark.sql.functions import col, unix_timestamp

# PySpark API
df.filter(col("PULocationID") == col("DOLocationID")) \
  .withColumn("trip_duration", 
              (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime"))/60) \
  .agg(avg("trip_duration").alias("average_duration_minutes")) \
  .show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT AVG(
    (UNIX_TIMESTAMP(tpep_dropoff_datetime) - UNIX_TIMESTAMP(tpep_pickup_datetime))/60
  ) AS average_duration_minutes
  FROM taxi_trips
  WHERE PULocationID = DOLocationID
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 9. Qual a proporção de viagens com apenas um passageiro em relação ao total de viagens?

# COMMAND ----------

# PySpark API
df.withColumn("single_passenger", (col("passenger_count") == 1).cast("int")) \
  .agg(avg("single_passenger").alias("proportion_single_passenger")) \
  .show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT AVG(
    CASE WHEN passenger_count = 1 THEN 1 ELSE 0 END
  ) AS proportion_single_passenger
  FROM taxi_trips
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 10. Quais são os top 3 horários de pico para início das corridas?

# COMMAND ----------

from pyspark.sql.functions import count

# PySpark API
df.withColumn("hour", hour(df.tpep_pickup_datetime)) \
  .groupBy("hour") \
  .agg(count("*").alias("number_of_trips")) \
  .orderBy("number_of_trips", ascending=False) \
  .limit(3).show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT HOUR(tpep_pickup_datetime) AS hour, 
         COUNT(*) AS number_of_trips
  FROM taxi_trips
  GROUP BY hour
  ORDER BY number_of_trips DESC
  LIMIT 3
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 11. Qual é a variação percentual na quantidade de viagens entre dias úteis e fins de semana?

# COMMAND ----------

from pyspark.sql.functions import dayofweek, when, count

# PySpark API
df.withColumn("is_weekend", when(dayofweek(df.tpep_pickup_datetime).isin(1, 7), 1).otherwise(0)) \
  .groupBy("is_weekend") \
  .agg(count("*").alias("number_of_trips")) \
  .withColumn("percent", col("number_of_trips") * 100 / sum("number_of_trips").over()) \
  .show()


# COMMAND ----------

# SQL
spark.sql("""
  SELECT is_weekend,
         COUNT(*) AS number_of_trips,
         COUNT(*) * 100 / SUM(COUNT(*)) OVER () AS percent
  FROM (SELECT *, CASE WHEN DAYOFWEEK(tpep_pickup_datetime) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend
        FROM taxi_trips)
  GROUP BY is_weekend
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 12. Quais são os 5 destinos mais comuns nas corridas que excedem $50?

# COMMAND ----------

# PySpark API
df.filter(df.total_amount > 50) \
  .groupBy("DOLocationID") \
  .agg(count("*").alias("number_of_trips")) \
  .orderBy("number_of_trips", ascending=False) \
  .limit(5) \
  .show()


# COMMAND ----------

# SQL
spark.sql("""
  SELECT DOLocationID, 
         COUNT(*) AS number_of_trips
  FROM taxi_trips
  WHERE total_amount > 50
  GROUP BY DOLocationID
  ORDER BY number_of_trips DESC
  LIMIT 5
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 13. Como a distância média e o valor médio da gorjeta estão relacionados?

# COMMAND ----------

from pyspark.sql.functions import avg

# PySpark API
df.groupBy("trip_distance") \
  .agg(avg("tip_amount").alias("average_tip")) \
  .orderBy("trip_distance", ascending=False) \
  .show()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT trip_distance, 
         AVG(tip_amount) AS average_tip
  FROM taxi_trips
  GROUP BY trip_distance
  ORDER BY trip_distance
""").show()


# COMMAND ----------

# MAGIC %md
# MAGIC #### Pergunta 14. Qual é o tempo médio de viagem entre as localizações mais populares de embarque e desembarque?

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# PySpark API
windowSpec  = Window.partitionBy("PULocationID", "DOLocationID")

df.withColumn("trip_duration", 
              (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime"))/60) \
  .withColumn("average_trip_duration", avg("trip_duration").over(windowSpec)) \
  .withColumn("diff_trip_duration_and_average", F.col("trip_duration") - F.col("average_trip_duration")) \
  .orderBy("diff_trip_duration_and_average", ascending=False) \
  .display()

# COMMAND ----------

# SQL
spark.sql("""
  SELECT PULocationID, 
         DOLocationID, 
         AVG((UNIX_TIMESTAMP(tpep_dropoff_datetime) - UNIX_TIMESTAMP(tpep_pickup_datetime))/60) 
         OVER (PARTITION BY PULocationID, DOLocationID) AS average_trip_duration
  FROM taxi_trips
  ORDER BY average_trip_duration DESC
""").show()

