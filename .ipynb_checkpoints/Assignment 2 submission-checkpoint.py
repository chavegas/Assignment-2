# Import all necessary libraries and setup the environment for matplotlib
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Normalizer, VectorAssembler
from pyspark.ml.linalg import *
from pyspark.sql.types import * 
from pyspark.sql.functions import *
import pyspark.sql.functions as f
import matplotlib.pyplot as plt

def blank_as_null(x):
    return when(col(x) != "", col(x)).otherwise(None)
spark = SparkSession \
    .builder \
    .appName("Assignment 2") \
    .getOrCreate()
    
tweeets_data = spark.read.option('multiline','true').json('tweets.json')    
tweets_agg = tweeets_data.groupby("user_id").agg(f.concat_ws(" ", f.collect_list(tweeets_data.id)).alias('agg_tweets'),
                                    f.concat_ws(" ", f.collect_list(tweeets_data.retweet_user_id)).alias('agg_retweet_users'),
                                    f.concat_ws(" ", f.collect_list(tweeets_data.retweet_id)).alias('agg_retweets'),
                                    f.concat_ws(" ", f.collect_list(tweeets_data.replyto_user_id)).alias('agg_reply_users'),
                                    f.concat_ws(" ", f.collect_list(tweeets_data.replyto_id)).alias('agg_replies'))
tweets_agg = tweets_agg.withColumn("agg_reply_users", blank_as_null("agg_reply_users"))
tweets_agg = tweets_agg.withColumn("agg_retweet_users", blank_as_null("agg_retweet_users"))
tweets_agg = tweets_agg.withColumn("agg_retweets", blank_as_null("agg_retweets"))
tweets_agg = tweets_agg.withColumn("agg_replies", blank_as_null("agg_replies"))                                    

tweets_processed = tweets_agg.select('*',concat_ws(' ','agg_retweets','agg_replies').alias('agg_tweet_respond'))

tokenizer = Tokenizer(inputCol='agg_tweet_respond',
    outputCol="vectors")
tweets_vectors = tokenizer.transform(tweets_processed)
hashingTF = HashingTF(inputCol="vectors", outputCol="tf")
tf = hashingTF.transform(tweets_vectors)

idf = IDF(inputCol="tf", outputCol="feature").fit(tf)
tfidf = idf.transform(tf)

normalizer = Normalizer(inputCol="feature", outputCol="norm")
tf_data = normalizer.transform(tfidf)

selected_id = 202170318
tweets_user_filtered = tf_data.where(f'user_id = {selected_id}')
compare_vector = tweets_user_filtered.first()['norm']

def cos_sim(a,b=compare_vector):
    return float(a.dot(b) / (a.norm(2) * b.norm(2)))
cos_function = udf(cos_sim, FloatType())

tf_data = tf_data.withColumn("CosineSim",cos_function('norm'))
tf_data = tf_data.where(f'user_id <> {selected_id}')
sorted_output_tf = tf_data.filter(tf_data.CosineSim > 0).sort(col('CosineSim').desc())
sorted_output_tf.show(5,truncate=False)

from pyspark.ml.feature import CountVectorizer

cv = CountVectorizer(inputCol="vectors", outputCol="features")

tweets_cv = cv.fit(tweets_vectors)

cv_data = tweets_cv.transform(tweets_vectors)
selected_id = 202170318
tweets_user_filtered = cv_data.where(f'user_id = {selected_id}')
compare_vector = tweets_user_filtered.first()['features']
#def cos_sim(a,b=compare_vector):
#    return float(a.dot(b) / (a.norm(2) * b.norm(2)))
cos_function = udf(cos_sim, FloatType())
cv_data = cv_data.withColumn("CosineSim",cos_function('features'))
cv_data = cv_data.where(f'user_id <> {selected_id}')
sorted_output_cv = cv_data.filter(cv_data.CosineSim > 0).sort(col('CosineSim').desc())
sorted_output_cv.show(5,truncate=False)