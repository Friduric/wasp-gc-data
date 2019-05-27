import pickle

from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.linalg.distributed import RowMatrix

import time

# Load matrix properties
with open('mycielskian11.pickle', 'rb') as file:
    matrix = pickle.load(file)
print("Pickled")

# Start Spark session
spark = SparkSession \
    .builder \
    .appName("PythonSvdPca") \
    .getOrCreate()

# Create Spark row matrix based on matrix properties
a = 1535
data = matrix[0]
ir = matrix[1]
jc = matrix[2]

sparse_matrix = Matrices.sparse(a, a, jc, ir, data)
A = sparse_matrix.toArray()
dist_data = spark.sparkContext.parallelize(A)
mat = RowMatrix(dist_data)

# Perform SVD and PCA computations oin the matrix
svd_times = []
pca_times = []

# Run svd computations
for i in range(10):
    start = time.time()
    svd = mat.computeSVD(5, computeU=True)
    stop = time.time()
    delta = stop - start
    svd_times.append(delta)

# Run pca computations
for i in range(10):
    start = time.time()
    pca = mat.computePrincipalComponents(4)
    stop = time.time()
    delta = stop - start
    pca_times.append(delta)

print("Execution times svd:")
print(svd_times)
print("Execution times pca:")
print(pca_times)

spark.stop()
