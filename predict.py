import pyspark
from pyspark import SparkContext,SparkConf

from pyspark.sql import SQLContext

from pyspark.sql import Row
from pyspark.sql import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer,VectorIndexer
from pyspark.ml.feature import StandardScaler,StandardScalerModel
from pyspark.ml.feature import ElementwiseProduct
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans

import numpy as np
from numpy import allclose
import matplotlib.pylab as plt

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import math as mat
import pickle

APP_NAME = "predict"

# In[9]:

# Calculate 'num_neighbours' nearest neighbours using KNN algorithm using 'tree_type'
# (Ball_tress, Kd_trees etc.)
# Returns recall and population shared as a tuple

def NN_helper_client(knn_train, knn_test, test_pd, num_neighbours, num_indices, tree_type):
    print("NN Helper start")
    # Initiating nearest neighbor search
    neighbors = NearestNeighbors(n_neighbors = num_neighbours,algorithm = tree_type).fit(knn_test)
    
    # Generating indices and distances for K Nearest Neighbours
    knn_distances,knn_indices = neighbors.kneighbors(knn_train)
    
    knn_indices_pd = pd.DataFrame(knn_indices.flatten(),columns=["indices"])

    # Calculating frequency for each instance in the KNN matrix
    freq_table = pd.crosstab(index = knn_indices_pd["indices"],columns = "freq")
    
    freq_table['ind'] = freq_table.index
    
    freq_table = freq_table.merge(test_pd[['cust_id','indices']],left_on="ind",right_on="indices",how="left")
    
    freq_table = freq_table.drop(['ind','indices'],axis=1)
    
    freq_table = freq_table.sort(["freq"], ascending = [0])
    
    freq_table = freq_table.head(num_indices)
    
    return (freq_table)


# In[10]:

def findNearestNeighbour_client(train_all,test_all,num_neighbours,num_indices,tree_type):
    
    # Fetching indices for training and test sets
    train_all['indices'] = train_all.index
    test_all['indices'] = test_all.index
    
    # Converting features into SparseVector format
    l_train=list()
    for k in train_all.scaled_weighted_features:
        l_train.append(Vectors.sparse(len(k), [(i,j) for i,j in enumerate(k)  if j != 0 ]))

    l_test=list()
    for k in test_all.scaled_weighted_features:
        l_test.append(Vectors.sparse(len(k), [(i,j) for i,j in enumerate(k)  if j != 0 ]))
    
    # Converting to a numpy array
    knn_train = np.asarray(l_train)
    knn_test = np.asarray(l_test)
    
    # Calling the helper function to get the frequency table 
    result = NN_helper_client(knn_train,knn_test,test_all,num_neighbours,num_indices,tree_type)
    
    return (result)

def main(sc):
    sqlContext = SQLContext(sc)
    input_path = ''
    output_path = ''
    model_path = ''
    model_info_path = model_path + ''
    model_scaler_path = model_path + ''
    model_train_set_path = model_path + ''

    #IMPORT THE CLIENT DATA
    client_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(input_path)

    # Load the models and train data from Training Interface paths
    model_info = sc.pickleFile(model_info_path).flatMap(lambda x: x.items()).collectAsMap()
    scalerModel = StandardScalerModel.load(model_scaler_path)
    df_master_new = sc.pickleFile(model_train_set_path).toDF()

    col_names = model_info['col_names']
    sorted_top_varimp = model_info['varimp']

    # Pulling data for most import 6 features
    client_data = client_data.select(client_data.u_msisdn.cast('string'),*(col(c).cast("double").alias(c) for c in col_names))

    # Defines that the first column is the unique ID, the last one contains the labels and all the ones in between are the given features
    client_master = client_data.rdd.map(lambda r: Row(cust_id=r[0],features=Vectors.dense(r[1:]))).toDF()

    # Scale and normaize the features so that all features can be compared
    # and create a new column for the features
    client_scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withStd=True, withMean=True)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = client_scaler.fit(client_master)

    # Normalize each feature to have unit standard deviation.
    client_master = scalerModel.transform(client_master)

    #The old features have been replaced with their scaled versions and thus
    # we no longer care about the old, unbalanced features
    client_master = client_master.drop('features')
    
    sqlContext.registerDataFrameAsTable(df_master_new,"df_master_train_table")

    # Remove the negative labels as only the positive ones are important
    train_all_client = sqlContext.sql('select * from df_master_train_table where label = 1')

    # Multiply feature values with corresponding importances
    m = ElementwiseProduct(scalingVec=Vectors.dense(sorted_top_varimp), inputCol="scaled_features", outputCol="scaled_weighted_features")

    train_all_client = m.transform(train_all_client)

    client_master = m.transform(client_master)

    sqlContext.dropTempTable("df_master_train_table")

    nn = 1000
    popshared = 0.30
    num_indices = (int)(popshared * client_master.count())
    tree_type = "kd_tree"
    nn,popshared,num_indices

    train_pd = train_all_client.toPandas()
    test_pd = client_master.toPandas()

    freq_table = findNearestNeighbour_client(train_pd,test_pd,nn,num_indices,tree_type)

    sqlContext.createDataFrame(freq_table[['cust_id','freq']],).repartition(1).write.format("com.databricks.spark.csv").save(output_path)

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("yarn")
    sc = SparkContext(conf=conf)
    main(sc)
