from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, Normalizer, VectorAssembler
from pyspark.ml.linalg import *
from pyspark.sql.types import * 
from pyspark.sql.functions import *
from pyspark.sql.functions import array, col, lit, struct
import pyspark.sql.functions as f
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer, IndexToString

def blank_as_null(x):
    return when(col(x) != "", col(x)).otherwise(None)
spark = SparkSession \
    .builder \
    .appName("Assignment 2 WL 2") \
    .getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions",100)    
tweeets_data = spark.read.option('multiline','true').json('tweets.json')  
wl2 = tweeets_data.withColumn("mentioned_users", tweeets_data["user_mentions"].getField('id')).cache()
wl2_users = wl2.select(col('user_id'),col('mentioned_users'))
wl2_users = wl2_users.withColumn("mentioned_users", explode("mentioned_users"))
wl2_users_agg = wl2_users.groupBy(col('user_id'),col('mentioned_users')).count().cache()

stringIndexer_uid = StringIndexer(inputCol="user_id", outputCol="user_id_indexed",stringOrderType="frequencyDesc")
model_uid = stringIndexer_uid.fit(wl2_users_agg)

stringIndexer_mentioned = StringIndexer(inputCol="mentioned_users", outputCol="mentioned_users_indexed",stringOrderType="frequencyDesc")
model_mu = stringIndexer_mentioned.fit(wl2_users_agg)

td = model_uid.transform(wl2_users_agg)
wl2_users_transformed = model_mu.transform(td)

als = ALS(userCol="user_id_indexed", itemCol="mentioned_users_indexed", ratingCol="count",
          coldStartStrategy="drop")
model = als.fit(wl2_users_transformed)

model_recs = model.recommendForAllUsers(5).cache()

uid_labels = model_uid.labels
uid_labels_ = array(*[lit(x) for x in uid_labels])

n = 5
mu_labels = model_mu.labels
mu_labels_ = array(*[lit(x) for x in mu_labels])
recommendations = array(*[struct(
    mu_labels_[col("recommendations")[i]["mentioned_users_indexed"]].alias("userId"),
    col("recommendations")[i]["rating"].alias("rating")
) for i in range(n)])

model_recs = model_recs.withColumn("recommendations", recommendations)\
        .withColumn("user_id", uid_labels_[col("user_id_indexed")])
model_recs.select(['user_id'] + [col("recommendations")[i].alias('Recommended User: '+str(i+1)) for i in range(n)]).show(truncate=False)