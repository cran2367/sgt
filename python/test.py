import pandas as pd
protein_data=pd.read_csv('../data/protein_classification.csv')
X=protein_data['Sequence']
def split(word): 
      return [char for char in word] 

sequences = [split(x) for x in X]
protein_data=pd.read_csv('../data/protein_classification.csv')
X=protein_data['Sequence']

import string
# import sgtdev as sgt
import sgt
# Spark
from pyspark import SparkContext
sc = SparkContext("local", "app")
rdd = sc.parallelize(sequences)
sgt_sc = sgt.Sgt(kappa = 1, lengthsensitive = False, mode="spark", alphabets=list(string.ascii_uppercase))
rdd_embedding = sgt_sc.fit_transform(corpus=rdd)

sc.stop()
# Multi-processing
sgt_mp = sgt.Sgt(kappa = 1, lengthsensitive = False, mode="multiprocessing", processors=3)
mp_embedding = sgt_mp.fit_transform(corpus=sequences)
mp_embedding = sgt_mp.transform(corpus=sequences)
# Default
sgt = sgt.Sgt(kappa = 1, lengthsensitive = False)
embedding = sgt.fit_transform(corpus=sequences)

# Spark again
corpus = [["B","B","A","C","A","C","A","A","B","A"], ["C", "Z", "Z", "Z", "D"]]

sc = SparkContext("local", "app")

rdd = sc.parallelize(corpus)

sgt_sc = sgt.Sgt(kappa = 1, 
                 lengthsensitive = False, 
                 mode="spark", 
                 alphabets=["A", "B", "C", "D", "Z"],
                 lazy=False)
s = sgt_sc.fit_transform(corpus=rdd)
print(s)
sc.stop()