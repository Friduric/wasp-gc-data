import h5py

#from pyspark import SparkContext
#from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import RowMatrix

import time

file_name = 'mycielskian3.mat'
f = h5py.File(file_name, 'r')

if file_name == 'mycielskian3.mat':
    a = 5
else:
    a = 767

spark = SparkSession \
    .builder \
    .appName("PythonPi") \
    .getOrCreate()

group = f['Problem']
data = group['A']['data'][()]
ir = group['A']['ir'][()]
jc = group['A']['jc'][()]

sparse_matrix = Matrices.sparse(a, a, jc, ir, data)
A = sparse_matrix.toArray()
dist_data = spark.sparkContext.parallelize(A)
mat = RowMatrix(dist_data)

start = time.time()

# Compute the top 4 principal components.
# Principal components are stored in a local dense matrix.
pc = mat.computePrincipalComponents(4)

# Project the rows to the linear space spanned by the top 4 principal components.
#projected = mat.multiply(pc)
# $example off$
#collected = projected.rows.collect()
#print("Projected Row Matrix of principal component:")
#for vector in collected:
#    print(vector)

stop = time.time()

delta = stop - start

f = open("spark_pca_execution_time.txt", "a")
f.write("Execution time: %.2f" % delta)
f.close()
print("Execution time: %.2f" % delta)


spark.stop()
