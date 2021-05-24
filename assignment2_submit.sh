export HADOOP_CONF_DIR=$HADOOP_HOME%/etc/hadoop
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 3 \
    Assignment\ 2\ Workload\ Combined.py \
    --output $1 
