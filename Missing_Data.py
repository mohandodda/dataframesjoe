# Databricks notebook source
# MAGIC %md # Missing Data
# MAGIC 
# MAGIC Often data sources are incomplete, which means you will have missing data, you have 3 basic options for filling in missing data (you will personally have to make the decision for what is the right approach:
# MAGIC 
# MAGIC * Just keep the missing data points.
# MAGIC * Drop them missing data points (including the entire row)
# MAGIC * Fill them in with some other value.
# MAGIC 
# MAGIC Let's cover examples of each of these methods!

# COMMAND ----------

# MAGIC %md ## Keeping the missing data
# MAGIC A few machine learning algorithms can easily deal with missing data, let's see what it looks like:

# COMMAND ----------

from pyspark.sql import SparkSession
# May take a little while on a local computer
spark = SparkSession.builder.appName("missingdata").getOrCreate()

# COMMAND ----------

df = spark.read.csv("ContainsNull.csv",header=True,inferSchema=True)

# COMMAND ----------

df.show()

# COMMAND ----------

# MAGIC %md Notice how the data remains as a null.

# COMMAND ----------

# MAGIC %md ## Drop the missing data
# MAGIC 
# MAGIC You can use the .na functions for missing data. The drop command has the following parameters:
# MAGIC 
# MAGIC     df.na.drop(how='any', thresh=None, subset=None)
# MAGIC     
# MAGIC     * param how: 'any' or 'all'.
# MAGIC     
# MAGIC         If 'any', drop a row if it contains any nulls.
# MAGIC         If 'all', drop a row only if all its values are null.
# MAGIC     
# MAGIC     * param thresh: int, default None
# MAGIC     
# MAGIC         If specified, drop rows that have less than `thresh` non-null values.
# MAGIC         This overwrites the `how` parameter.
# MAGIC         
# MAGIC     * param subset: 
# MAGIC         optional list of column names to consider.

# COMMAND ----------

# Drop any row that contains missing data
df.na.drop().show()

# COMMAND ----------

# Has to have at least 2 NON-null values
df.na.drop(thresh=2).show()

# COMMAND ----------

df.na.drop(subset=["Sales"]).show()

# COMMAND ----------

df.na.drop(how='any').show()

# COMMAND ----------

df.na.drop(how='all').show()

# COMMAND ----------

# MAGIC %md ## Fill the missing values
# MAGIC 
# MAGIC We can also fill the missing values with new values. If you have multiple nulls across multiple data types, Spark is actually smart enough to match up the data types. For example:

# COMMAND ----------

df.na.fill('NEW VALUE').show()

# COMMAND ----------

df.na.fill(0).show()

# COMMAND ----------

# MAGIC %md Usually you should specify what columns you want to fill with the subset parameter

# COMMAND ----------

df.na.fill('No Name',subset=['Name']).show()

# COMMAND ----------

# MAGIC %md A very common practice is to fill values with the mean value for the column, for example:

# COMMAND ----------

from pyspark.sql.functions import mean
mean_val = df.select(mean(df['Sales'])).collect()

# Weird nested formatting of Row object!
mean_val[0][0]

# COMMAND ----------

mean_sales = mean_val[0][0]

# COMMAND ----------

df.na.fill(mean_sales,["Sales"]).show()

# COMMAND ----------

# One (very ugly) one-liner
df.na.fill(df.select(mean(df['Sales'])).collect()[0][0],['Sales']).show()

# COMMAND ----------

# MAGIC %md That is all we need to know for now!
