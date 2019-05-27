#import h5py

import pickle

#from pyspark import SparkContext
#from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import RowMatrix

import time

with open('mycielskian10.pickle', 'rb') as file:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    matrix = pickle.load(file, protocol=2)

print("Pickled")

a = 767
#a = 3

spark = SparkSession \
    .builder \
    .appName("PythonPi") \
    .getOrCreate()

data = matrix[0]
ir = matrix[1]
jc = matrix[2]

sparse_matrix = Matrices.sparse(a, a, jc, ir, data)
A = sparse_matrix.toArray()
dist_data = spark.sparkContext.parallelize(A)
mat = RowMatrix(dist_data)

start = time.time()
svd = mat.computeSVD(5, computeU=True)
stop = time.time()

delta = stop - start

f = open("spark_svd_execution_time.txt", "a")
f.write("Execution time: %.2f" % delta)
f.close()
print("Execution time: %.2f" % delta)

#U = svd.U
#s = svd.s
#V = svd.V
#collected = U.rows.collect()

spark.stop()
