# In[2]:

import pyspark
from pyspark import SparkContext,SparkConf

from pyspark.sql import SQLContext

from pyspark.sql import Row
from pyspark.sql import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorIndexer
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

APP_NAME = "train"

# Calculate 'num_neighbours' nearest neighbours using KNN algorithm using 'tree_type'
# (Ball_tress, Kd_trees etc.)
# Returns recall and population shared as a tuple

def NN_helper(knn_train, knn_test, test_df, num_neighbours, num_indices, tree_type):
    
    # Initiating nearest neighbor search
    neighbors = NearestNeighbors(n_neighbors = num_neighbours,algorithm = tree_type).fit(knn_test)
    
    # Generating indices and distances for K Nearest Neighbours
    knn_distances,knn_indices = neighbors.kneighbors(knn_train)
    
    knn_indices_pd = pd.DataFrame(knn_indices.flatten(),columns=["indices"])

    # Calculating frequency for each instance in the KNN matrix
    freq_table = pd.crosstab(index = knn_indices_pd["indices"],columns = "freq")
    
    freq_table = freq_table.sort(["freq"], ascending = [0])
    
    # Selecting instances from test set with frequency greater than cutoff
    selected_lookalike_indices = freq_table.head(num_indices).index.tolist()
    
    # Subsetting the test set for data on the selected instances
    selected_lookalikes = test_df[test_df['indices'].isin(selected_lookalike_indices)]
    
    positive_lookalikes = selected_lookalikes[selected_lookalikes['label'] == '1']

    all_positives = test_df[test_df['label'] == '1']
    
    # Calculating the number of instances for each set
    num_positive_lookalikes = (float)(len(positive_lookalikes))
    num_all_positives = (float)(len(all_positives))
    num_selected_lookalikes = (float)(len(selected_lookalike_indices))
    num_test_instances = (float)(len(test_df))
    
    # Calculating recall and population shared
    # Ratio of individuals labeled positive by the algorithm to the actual number of positivies
    recall = num_positive_lookalikes/num_all_positives
    
    return recall

# Format the training and test sets to call K Nearest Neighbours algorithm

def findNearestNeighbour(train_broadcast,test_broadcast,num_neighbours,num_indices,test_all,tree_type):
         
    # Fetching the training and test dataframes
    knn_train = train_broadcast.value
    knn_test = test_broadcast.value
    
    # Calling the helper function to 
    recall = NN_helper(knn_train,knn_test,test_all,num_neighbours,num_indices,tree_type)
    
    return (recall, num_neighbours)

def main(sc):
    sqlContext = SQLContext(sc)
    # In[1]:
    input_path = ''
    model_path = ''
    model_info_path = model_path + ''
    model_scaler_path = model_path + ''
    model_train_set_path = model_path + ''

    # Import the table of features and labels into dataframes
    df_data = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(input_path)

    # Convert all features to double type except for ID and Label, which remain as strings
    # This is done because the Random Forest Algorithm requires features to be numbers
    df_data = df_data.select(*(col(c).cast("double").alias(c) for c in df_data.columns[1:-1]),df_data.u_msisdn.cast('string'),df_data.tag.cast('string'))

    # Defines that the first column is the unique ID, the last one contains the labels and all the ones in between are the given features
    df_master = df_data.rdd.map(lambda r: Row(cust_id=r[-2],label=r[-1],features=Vectors.dense(r[:-2]))).toDF()

    # Randomly Split the data into a test and train set
    (df_master_train, df_master_test) = df_master.randomSplit([0.5, 0.5], seed = 123)

    # Set the Random Forest input to the training set
    rf_init_data = df_master_train

    # Indexing labels for Random Forest Algorithm
    labelIndexer = StringIndexer(inputCol="label",outputCol="indexed_label")
    model = labelIndexer.fit(rf_init_data)
    rf_init_data = model.transform(rf_init_data)

    # Indexing features for Random Forest Algorithm
    featureIndexer = VectorIndexer(inputCol="features",outputCol="indexed_features",maxCategories=2)
    model = featureIndexer.fit(rf_init_data)
    rf_init_data = model.transform(rf_init_data)

    # Configures inbuilt Random Forest Classifier function with 500 trees,
    # max depth = 8 and 32 bins
    rf_init = RandomForestClassifier(labelCol="indexed_label", featuresCol="indexed_features",numTrees=500, impurity="gini",maxDepth=8, maxBins=32)

    rf_init_data.persist() # Cache the data set
    rf_init_model = rf_init.fit(rf_init_data) # Run the Random Forest Algorithm

    rf_init_data.unpersist()

    # Extract a list of feature importances from the output of the Random Forest
    # Algorithm with each element corresponding to a feature
    rf_init_varimp = np.sqrt(rf_init_model.featureImportances.toArray())

    # Creates a list containing the 6 most important features to be used later
    # to subset our entire data from 146 features to just 6!

    # Create a list containing the names of all features
    column_names = df_data.columns[:-2]

    #Creating a dictionary mapping feature names to their respective importances
    NameToImp = dict()
    for i in range(len(column_names)):
        key = column_names[i]
        value = rf_init_varimp[i]
        NameToImp[key] = value

    # Sorted list in reverse order according to the variable importances
    sorted_varimp = sorted(NameToImp.values(), reverse = True)

    # Collect importances of 6 most important features
    sorted_top_varimp = sorted_varimp[:6]

    # Sorted list of column names in reverse order according to varimp
    sorted_colnames = sorted(NameToImp,key=NameToImp.get,reverse=True)

    # Collect colnames of 6 most imp features
    col_names = sorted_colnames[:6]

    # Pulling data for most import 6 features
    df_data_new = df_data.select(df_data.u_msisdn.cast('string'), df_data.tag.cast('string'),*(col(c).cast("double").alias(c) for c in col_names))

    # Defines that the first column is the unique ID, the last one contains the labels and all the ones in between are the given features
    df_master_new = df_data_new.rdd.map(lambda r: Row(cust_id=r[0],label=r[1],features=Vectors.dense(r[2:]))).toDF()

    # Scale and normaize the features so that all features can be compared
    # and create a new column for the features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withStd=True, withMean=True)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(df_master_new)

    # Normalize each feature to have unit standard deviation.
    df_master_new = scalerModel.transform(df_master_new)

    #The old features have been replaced with their scaled versions and thus
    # we no longer care about the old, unbalanced features
    df_master_new = df_master_new.drop('features')

    # Randomly Split the data into a test and train set
    (df_master_train, df_master_test) = df_master_new.randomSplit([0.5, 0.5], seed = 123)

    test_all = df_master_test

    sqlContext.registerDataFrameAsTable(df_master_train,"df_master_train_table")

    # Remove the negative labels as only the positive ones are important
    train_all = sqlContext.sql('select * from df_master_train_table where label = 1')

    # Multiply feature values with corresponding importances
    m = ElementwiseProduct(scalingVec=Vectors.dense(sorted_top_varimp), inputCol="scaled_features", outputCol="scaled_weighted_features")

    train_all = m.transform(train_all)

    test_all = m.transform(test_all)

    sqlContext.dropTempTable("df_master_train_table")

    # Create a list of tasks containing tuples of number of neighbours and 
    # cutoff frequencies to be passed to KNN algorithm
    number_of_neighbours = [250,550,750,1000]
    popshared = 0.30
    num_indices = int(popshared * (test_all.count()))
    tasks = []
    for num_neighbour in number_of_neighbours:
        tasks = tasks + [(num_neighbour, num_indices)]

    # Partitioning the tasks for parallel processing
    tasksRDD = sc.parallelize(tasks,numSlices=len(tasks))
    tasksRDD.collect()

    train_pd = train_all.toPandas()
    test_pd = test_all.toPandas()

    train_pd['indices'] = train_pd.index
    test_pd['indices'] = test_pd.index

        # Converting features into SparseVector format
    l_train=list()
    for k in train_pd.scaled_weighted_features:
        l_train.append(Vectors.sparse(len(k), [(i,j) for i,j in enumerate(k)  if j != 0 ]))

    l_test=list()
    for k in test_pd.scaled_weighted_features:
        l_test.append(Vectors.sparse(len(k), [(i,j) for i,j in enumerate(k)  if j != 0 ]))

        # Converting to a numpy array
    knn_train = np.asarray(l_train)
    knn_test = np.asarray(l_test)
    # Broadcasting the training and test sets to all partitions
    train_broadcast = sc.broadcast(knn_train)
    test_broadcast = sc.broadcast(knn_test)

    # Calling K Nearest Neighbour search on each partition
    tree_type = "kd_tree"
    resultsRDD = tasksRDD.map(lambda nc: findNearestNeighbour(train_broadcast,test_broadcast,nc[0],nc[1],test_pd, tree_type))
    resultsRDD.cache()
    resultsRDD.count()

    resultsPD = resultsRDD.toDF().toPandas()

    resultsPD["popshared"] = popshared
    resultsPD=resultsPD.rename(columns = {'_1':'Recall'})
    resultsPD=resultsPD.rename(columns = {'_2':'Number of Neighbors'})

    bestResult = (resultsPD.sort_values(by=["Recall"],ascending = [0])).iloc[0]
    bestNN = int(bestResult["Number of Neighbors"])
    bestRecall = bestResult["Recall"]

    # saving the model info - varimp,recall,NN,col_names to model_path
    column_names = [i for i in col_names]
    model_info = sc.parallelize([{"varimp": sorted_top_varimp, "recall": bestRecall, "NN": bestNN, "col_names": column_names}])
    model_info.saveAsPickleFile(path=model_info_path)

    # saving the scaler model to model_path
    scalerModel.write().overwrite().save(model_scaler_path)

    # saving the train set to model_path
    df_master_new.rdd.saveAsPickleFile(path=model_train_set_path)	

if __name__ == "__main__":
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("yarn")
    sc = SparkContext(conf=conf)
    main(sc)
