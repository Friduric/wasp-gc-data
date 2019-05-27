import scipy.io as spio
import numpy as np
import h5py

from pyspark import SparkContext
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import RowMatrix

file_name = 'mycielskian10.mat'
f = h5py.File(file_name,'r')

print("spark version: ")
print(SparkContext.version)
print("\n\n\n")

spark = SparkSession\
        .builder\
        .appName("PythonPi")\
        .getOrCreate()

for key in f.keys():
	group = f[key]
	first_key = key
	for key in group.keys():
		if key == 'A':
			print("First key %s" % first_key)
			print(list(group[key]))
			data = group[key]['data'][()]
			ir = group[key]['ir'][()]
			jc = group[key]['jc'][()]
			print(data.shape)
			print(ir.shape)
			print(jc.shape)
			a = 0
			if file_name == 'mycielskian3.mat':
				a = 5
			else:
				a = 767
			sparse_matrix = Matrices.sparse(a, a, jc, ir, data)
			A = sparse_matrix.toArray()
			print(sparse_matrix.toArray())
			distdata = spark.sparkContext.parallelize(A)
			mat = RowMatrix(distdata)
			svd = mat.computeSVD(5, computeU=True)
			U = svd.U
			s = svd.s
			V = svd.V
			collected = U.rows.collect()
			print("U factor is:")
			for vector in collected:
				print(vector)
			print("Singular values are: %s" % s)
			print("V factor is: \n%s" % V)
        #for data in group[key]:
        #    print(data)
		
spark.stop()
		
#print(f)


#print(f.keys())
#print(list(f.keys()))
#a_group_key = list(f.keys())[1][0]

#data = list(f[a_group_key])

#data = list(f.get('Problem'))
#data = list(f.get('#refs#'))
#data = f['Problem'][0].value
#print(data)


# mat = spio.loadmat('mycielskian10.mat', squeeze_me=True)

# a = mat['a']
# S = mat['S']
# M = mat['M']

# print(mat)

# data = mat['Problem']
# print(data.dtype)
# print(data.shape)
# print(type(data))