# Databricks notebook source
# MAGIC %md # Spark DataFrames Project Exercise - SOLUTIONS

# COMMAND ----------

# MAGIC %md Let's get some quick practice with your new Spark DataFrame skills, you will be asked some basic questions about some stock market data, in this case Walmart Stock from the years 2012-2017. This exercise will just ask a bunch of questions, unlike the future machine learning exercises, which will be a little looser and be in the form of "Consulting Projects", but more on that later!
# MAGIC 
# MAGIC For now, just answer the questions and complete the tasks below.

# COMMAND ----------

# MAGIC %md #### Use the walmart_stock.csv file to Answer and complete the  tasks below!

# COMMAND ----------

# MAGIC %md #### Start a simple Spark Session

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("walmart").getOrCreate()

# COMMAND ----------

# MAGIC %md #### Load the Walmart Stock CSV File, have Spark infer the data types.

# COMMAND ----------

df = spark.read.csv('walmart_stock.csv',header=True,inferSchema=True)

# COMMAND ----------

# MAGIC %md #### What are the column names?

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md #### What does the Schema look like?

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md #### Print out the first 5 columns.

# COMMAND ----------

# Didn't strictly need a for loop, could have just then head()
for row in df.head(5):
    print(row)
    print('\n')

# COMMAND ----------

# MAGIC %md #### Use describe() to learn about the DataFrame.

# COMMAND ----------

df.describe().show()

# COMMAND ----------

# MAGIC %md ## Bonus Question!
# MAGIC #### There are too many decimal places for mean and stddev in the describe() dataframe. Format the numbers to just show up to two decimal places. Pay careful attention to the datatypes that .describe() returns, we didn't cover how to do this exact formatting, but we covered something very similar. [Check this link for a hint](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Column.cast)
# MAGIC 
# MAGIC If you get stuck on this, don't worry, just view the solutions.

# COMMAND ----------

# Uh oh Strings! 
df.describe().printSchema()

# COMMAND ----------

from pyspark.sql.functions import format_number

# COMMAND ----------

result = df.describe()
result.select(result['summary'],
              format_number(result['Open'].cast('float'),2).alias('Open'),
              format_number(result['High'].cast('float'),2).alias('High'),
              format_number(result['Low'].cast('float'),2).alias('Low'),
              format_number(result['Close'].cast('float'),2).alias('Close'),
              result['Volume'].cast('int').alias('Volume')
             ).show()

# COMMAND ----------

# MAGIC %md #### Create a new dataframe with a column called HV Ratio that is the ratio of the High Price versus volume of stock traded for a day.

# COMMAND ----------

df2 = df.withColumn("HV Ratio",df["High"]/df["Volume"])#.show()
# df2.show()
df2.select('HV Ratio').show()

# COMMAND ----------

# MAGIC %md #### What day had the Peak High in Price?

# COMMAND ----------

# Didn't need to really do this much indexing
# Could have just shown the entire row
df.orderBy(df["High"].desc()).head(1)[0][0]

# COMMAND ----------

# MAGIC %md #### What is the mean of the Close column?

# COMMAND ----------

# Also could have gotten this from describe()
from pyspark.sql.functions import mean
df.select(mean("Close")).show()

# COMMAND ----------

# MAGIC %md #### What is the max and min of the Volume column?

# COMMAND ----------

# Could have also used describe
from pyspark.sql.functions import max,min

# COMMAND ----------

df.select(max("Volume"),min("Volume")).show()

# COMMAND ----------

# MAGIC %md #### How many days was the Close lower than 60 dollars?

# COMMAND ----------

df.filter("Close < 60").count()

# COMMAND ----------

df.filter(df['Close'] < 60).count()

# COMMAND ----------

from pyspark.sql.functions import count
result = df.filter(df['Close'] < 60)
result.select(count('Close')).show()

# COMMAND ----------

# MAGIC %md #### What percentage of the time was the High greater than 80 dollars ?
# MAGIC #### In other words, (Number of Days High>80)/(Total Days in the dataset)

# COMMAND ----------

# 9.14 percent of the time it was over 80
# Many ways to do this
(df.filter(df["High"]>80).count()*1.0/df.count())*100

# COMMAND ----------

# MAGIC %md #### What is the Pearson correlation between High and Volume?
# MAGIC #### [Hint](http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameStatFunctions.corr)

# COMMAND ----------

from pyspark.sql.functions import corr
df.select(corr("High","Volume")).show()

# COMMAND ----------

# MAGIC %md #### What is the max High per year?

# COMMAND ----------

from pyspark.sql.functions import year
yeardf = df.withColumn("Year",year(df["Date"]))

# COMMAND ----------

max_df = yeardf.groupBy('Year').max()

# COMMAND ----------

# 2015
max_df.select('Year','max(High)').show()

# COMMAND ----------

# MAGIC %md #### What is the average Close for each Calendar Month?
# MAGIC #### In other words, across all the years, what is the average Close price for Jan,Feb, Mar, etc... Your result will have a value for each of these months. 

# COMMAND ----------

from pyspark.sql.functions import month
monthdf = df.withColumn("Month",month("Date"))
monthavgs = monthdf.select("Month","Close").groupBy("Month").mean()
monthavgs.select("Month","avg(Close)").orderBy('Month').show()

# COMMAND ----------

# MAGIC %md # Great Job!
