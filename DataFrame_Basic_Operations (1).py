# Databricks notebook source
# MAGIC %md # Basic Operations
# MAGIC 
# MAGIC This lecture will cover some basic operations with Spark DataFrames.
# MAGIC 
# MAGIC We will play around with some stock data from Apple.

# COMMAND ----------

from pyspark.sql import SparkSession

# COMMAND ----------

# May take awhile locally
spark = SparkSession.builder.appName("Operations").getOrCreate()

# COMMAND ----------

# Let Spark know about the header and infer the Schema types!
df = spark.read.csv('appl_stock.csv',inferSchema=True,header=True)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md ## Filtering Data
# MAGIC 
# MAGIC A large part of working with DataFrames is the ability to quickly filter out data based on conditions. Spark DataFrames are built on top of the Spark SQL platform, which means that is you already know SQL, you can quickly and easily grab that data using SQL commands, or using the DataFram methods (which is what we focus on in this course).

# COMMAND ----------

# Using SQL
df.filter("Close<500").show()

# COMMAND ----------

# Using SQL with .select()
df.filter("Close<500").select('Open').show()

# COMMAND ----------

# Using SQL with .select()
df.filter("Close<500").select(['Open','Close']).show()

# COMMAND ----------

# MAGIC %md Using normal python comparison operators is another way to do this, they will look very similar to SQL operators, except you need to make sure you are calling the entire column within the dataframe, using the format: df["column name"]
# MAGIC 
# MAGIC Let's see some examples:

# COMMAND ----------

df.filter(df["Close"] < 200).show()

# COMMAND ----------

# Will produce an error, make sure to read the error!
df.filter(df["Close"] < 200 and df['Open'] > 200).show()

# COMMAND ----------

# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) & (df['Open'] > 200) ).show()

# COMMAND ----------

# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) | (df['Open'] > 200) ).show()

# COMMAND ----------

# Make sure to add in the parenthesis separating the statements!
df.filter( (df["Close"] < 200) & ~(df['Open'] < 200) ).show()

# COMMAND ----------

df.filter(df["Low"] == 197.16).show()

# COMMAND ----------

# Collecting results as Python objects
df.filter(df["Low"] == 197.16).collect()

# COMMAND ----------

result = df.filter(df["Low"] == 197.16).collect()

# COMMAND ----------

# Note the nested structure returns a nested row object
type(result[0])

# COMMAND ----------

row = result[0]

# COMMAND ----------

# MAGIC %md Rows can be called to turn into dictionaries

# COMMAND ----------

row.asDict()

# COMMAND ----------

for item in result[0]:
    print(item)

# COMMAND ----------

# MAGIC %md That is all for now Great Job!
