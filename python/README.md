```python
# -*- coding: utf-8 -*-
# Authors: Chitta Ranjan <cran2367@gmail.com>
#
# License: BSD 3 clause
```

# Sgt definition.

## Purpose

Sequence Graph Transform (SGT) is a sequence embedding function. SGT extracts the short- and long-term sequence features and embeds them in a finite-dimensional feature space. With SGT you can tune the amount of short- to long-term patterns extracted in the embeddings without any  increase in the computation."

```
class Sgt():
    '''
    Compute embedding of a single or a collection of discrete item 
    sequences. A discrete item sequence is a sequence made from a set
    discrete elements, also known as alphabet set. For example,
    suppose the alphabet set is the set of roman letters, 
    {A, B, ..., Z}. This set is made of discrete elements. Examples of
    sequences from such a set are AABADDSA, UADSFJPFFFOIHOUGD, etc.
    Such sequence datasets are commonly found in online industry,
    for example, item purchase history, where the alphabet set is
    the set of all product items. Sequence datasets are abundant in
    bioinformatics as protein sequences.
    Using the embeddings created here, classification and clustering
    models can be built for sequence datasets.
    Read more in https://arxiv.org/pdf/1608.03533.pdf
    '''
```
    Parameters
    ----------
    Input:

    alphabets       Optional, except if mode is Spark. 
                    The set of alphabets that make up all 
                    the sequences in the dataset. If not passed, the
                    alphabet set is automatically computed as the 
                    unique set of elements that make all the sequences.
                    A list or 1d-array of the set of elements that make up the      
                    sequences. For example, np.array(["A", "B", "C"].
                    If mode is 'spark', the alphabets are necessary.

    kappa           Tuning parameter, kappa > 0, to change the extraction of 
                    long-term dependency. Higher the value the lesser
                    the long-term dependency captured in the embedding.
                    Typical values for kappa are 1, 5, 10.

    lengthsensitive Default false. This is set to true if the embedding of
                    should have the information of the length of the sequence.
                    If set to false then the embedding of two sequences with
                    similar pattern but different lengths will be the same.
                    lengthsensitive = false is similar to length-normalization.
                    
    flatten         Default True. If True the SGT embedding is flattened and returned as
                    a vector. Otherwise, it is returned as a matrix with the row and col
                    names same as the alphabets. The matrix form is used for            
                    interpretation purposes. Especially, to understand how the alphabets
                    are "related". Otherwise, for applying machine learning or deep
                    learning algorithms, the embedding vectors are required.
    
    mode            Choices in {'default', 'multiprocessing', 'spark'}.
    
    processors      Used if mode is 'multiprocessing'. By default, the 
                    number of processors used in multiprocessing is
                    number of available - 1.
    
    lazy            Used if mode is 'spark'. Default is False. If False,
                    the SGT embeddings are computed for each sequence
                    in the inputted RDD and returned as a list of 
                    embedding vectors. Otherwise, the RDD map is returned.
    '''

    Attributes
    ----------
    def fit(sequence)
    
    Extract Sequence Graph Transform features using Algorithm-2 in https://arxiv.org/abs/1608.03533.
    Input:
    sequence        An array of discrete elements. For example,
                    np.array(["B","B","A","C","A","C","A","A","B","A"].
                    
    Output: 
    sgt embedding   sgt matrix or vector (depending on Flatten==False or True) of the sequence
    
    
    --
    def fit_transform(corpus)
    
    Extract SGT embeddings for all sequences in a corpus. It finds
    the alphabets encompassing all the sequences in the corpus, if not inputted. 
    However, if the mode is 'spark', then the alphabets list has to be
    explicitly given in Sgt object declaration.
    
    Input:
    corpus          A list of sequences. Each sequence is a list of alphabets.
    
    Output:
    sgt embedding of all sequences in the corpus.
    
    
    --
    def transform(corpus)
    
    Find SGT embeddings of a new data sample belonging to the same population
    of the corpus that was fitted initially.

## Illustrative examples


```python
import numpy as np
import pandas as pd
from itertools import chain
import warnings

########
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics
import time

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(7) # fix random seed for reproducibility

from sgt import Sgt
```

    Using TensorFlow backend.


## Installation Test Examples


```python
# Learning a sgt embedding as a matrix with 
# rows and columns as the sequence alphabets. 
# This embedding shows the relationship between 
# the alphabets. The higher the value the 
# stronger the relationship.

sgt = Sgt(flatten=False)
sequence = np.array(["B","B","A","C","A","C","A","A","B","A"])
sgt.fit(sequence)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.906163</td>
      <td>1.310023</td>
      <td>2.618487</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.865694</td>
      <td>1.230423</td>
      <td>0.525440</td>
    </tr>
    <tr>
      <th>C</th>
      <td>1.371416</td>
      <td>0.282625</td>
      <td>1.353353</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Learning the sgt embeddings as vector for
# all sequences in a corpus.

sgt = Sgt(kappa=1, lengthsensitive=False)
corpus = [["B","B","A","C","A","C","A","A","B","A"], ["C", "Z", "Z", "Z", "D"]]

s = sgt.fit_transform(corpus)
print(s)
```

    [[0.90616284 1.31002279 2.6184865  0.         0.         0.86569371
      1.23042262 0.52543984 0.         0.         1.37141609 0.28262508
      1.35335283 0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.09157819 0.92166965 0.         0.         0.
      0.         0.         0.         0.         0.         0.92166965
      1.45182361]]



```python
# Change the parameters from default to
# a tuned value.

sgt = Sgt(kappa=5, lengthsensitive=True)
corpus = [["B","B","A","C","A","C","A","A","B","A"], ["C", "Z", "Z", "Z", "D"]]

s = sgt.fit_transform(corpus)
print(s)
```

    [[0.23305129 0.2791752  0.33922608 0.         0.         0.26177435
      0.29531212 0.10270374 0.         0.         0.28654051 0.04334255
      0.13533528 0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.01831564 0.29571168 0.         0.         0.
      0.         0.         0.         0.         0.         0.29571168
      0.3394528 ]]



```python
# Change the mode for faster computation.
# Mode: 'multiprocessing'
# Uses the multiple processors (CPUs) avalaible.

corpus = [["B","B","A","C","A","C","A","A","B","A"], ["C", "Z", "Z", "Z", "D"]]

sgt = Sgt(mode='multiprocessing')
s = sgt.fit_transform(corpus)
print(s)
```

    [[0.90616284 1.31002279 2.6184865  0.         0.         0.86569371
      1.23042262 0.52543984 0.         0.         1.37141609 0.28262508
      1.35335283 0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.        ]
     [0.         0.         0.         0.         0.         0.
      0.         0.         0.         0.         0.         0.
      0.         0.09157819 0.92166965 0.         0.         0.
      0.         0.         0.         0.         0.         0.92166965
      1.45182361]]



```python
# Change the mode for faster computation.
# Mode: 'spark'
# Uses spark RDD.

from pyspark import SparkContext
sc = SparkContext("local", "app")

corpus = [["B","B","A","C","A","C","A","A","B","A"], ["C", "Z", "Z", "Z", "D"]]

rdd = sc.parallelize(corpus)

sgt_sc = sgt.Sgt(kappa = 1, 
                 lengthsensitive = False, 
                 mode="spark", 
                 alphabets=["A", "B", "C", "D", "Z"],
                 lazy=False)

s = sgt_sc.fit_transform(corpus=rdd)

print(s)
```

# Real data examples

## Protein Sequence Data Analysis

The data used here is taken from www.uniprot.org. This is a public database for proteins. The data contains the protein sequences and their functions. In the following, we will demonstrate 
- clustering of the sequences.
- classification of the sequences with the functions as labels.


```python
protein_data=pd.read_csv('../data/protein_classification.csv')
X=protein_data['Sequence']
def split(word): 
    return [char for char in word] 

sequences = [split(x) for x in X]
print(sequences[0])
```

    ['M', 'E', 'I', 'E', 'K', 'T', 'N', 'R', 'M', 'N', 'A', 'L', 'F', 'E', 'F', 'Y', 'A', 'A', 'L', 'L', 'T', 'D', 'K', 'Q', 'M', 'N', 'Y', 'I', 'E', 'L', 'Y', 'Y', 'A', 'D', 'D', 'Y', 'S', 'L', 'A', 'E', 'I', 'A', 'E', 'E', 'F', 'G', 'V', 'S', 'R', 'Q', 'A', 'V', 'Y', 'D', 'N', 'I', 'K', 'R', 'T', 'E', 'K', 'I', 'L', 'E', 'D', 'Y', 'E', 'M', 'K', 'L', 'H', 'M', 'Y', 'S', 'D', 'Y', 'I', 'V', 'R', 'S', 'Q', 'I', 'F', 'D', 'Q', 'I', 'L', 'E', 'R', 'Y', 'P', 'K', 'D', 'D', 'F', 'L', 'Q', 'E', 'Q', 'I', 'E', 'I', 'L', 'T', 'S', 'I', 'D', 'N', 'R', 'E']


### Generating sequence embeddings


```python
sgt = Sgt(kappa=1, lengthsensitive=False, mode='multiprocessing')
```


```python
%%time
embedding = sgt.fit_transform(corpus=sequences)
```

    CPU times: user 79.5 ms, sys: 46 ms, total: 125 ms
    Wall time: 6.61 s



```python
embedding.shape
```




    (2112, 400)



#### Sequence Clustering
We perform PCA on the sequence embeddings and then do kmeans clustering.


```python
pca = PCA(n_components=2)
pca.fit(embedding)
X=pca.transform(embedding)

print(np.sum(pca.explained_variance_ratio_))
df = pd.DataFrame(data=X, columns=['x1', 'x2'])
df.head()
```

    0.6432744907364913





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.384913</td>
      <td>-0.269873</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.022764</td>
      <td>0.135995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.177792</td>
      <td>-0.172454</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.168074</td>
      <td>-0.147334</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.383616</td>
      <td>-0.271163</td>
    </tr>
  </tbody>
</table>
</div>




```python
kmeans = KMeans(n_clusters=3, max_iter =300)
kmeans.fit(df)

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}
colors = list(map(lambda x: colmap[x+1], labels))
plt.scatter(df['x1'], df['x2'], color=colors, alpha=0.5, edgecolor=colors)
```




    <matplotlib.collections.PathCollection at 0x13bd97438>




![png](output_19_1.png)


#### Sequence Classification
We perform PCA on the sequence embeddings and then do kmeans clustering.


```python
y = protein_data['Function [CC]']
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
```

We will perform a 10-fold cross-validation to measure the performance of the classification model.


```python
kfold = 10
X = pd.DataFrame(embedding)
y = encoded_y

random_state = 1

test_F1 = np.zeros(kfold)
skf = KFold(n_splits = kfold, shuffle = True, random_state = random_state)
k = 0
epochs = 50
batch_size = 128

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Sequential()
    model.add(Dense(64, input_shape = (X_train.shape[1],))) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train ,batch_size=batch_size, epochs=epochs, verbose=0)
    
    y_pred = model.predict_proba(X_test).round().astype(int)
    y_train_pred = model.predict_proba(X_train).round().astype(int)

    test_F1[k] = sklearn.metrics.f1_score(y_test, y_pred)
    k+=1
    
print ('Average f1 score', np.mean(test_F1))
```

    Average f1 score 1.0


## Weblog Data Analysis
This data sample is taken from https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset. 
This is a network intrusion data containing audit logs and any attack as a positive label. Since, network intrusion is a rare event, the data is unbalanced. Here we will,
- build a sequence classification model to predict a network intrusion.

Each sequence contains in the data is a series of activity, for example, {login, password}. The _alphabets_ in the input data sequences are already encoded into integers. The original sequences data file is also present in the `/data` directory.


```python
darpa_data = pd.read_csv('../data/darpa_data.csv')
darpa_data.columns
```




    Index(['timeduration', 'seqlen', 'seq', 'class'], dtype='object')




```python
X = darpa_data['seq']
sequences = [x.split('~') for x in X]
```


```python
y = darpa_data['class']
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
```

### Generating sequence embeddings
In this data, the sequence embeddings should be length-sensitive. The lengths are important here because sequences with similar patterns but different lengths can have different labels. Consider a simple example of two sessions: `{login, pswd, login, pswd,...}` and `{login, pswd,...(repeated several times)..., login, pswd}`. While the first session can be a regular user mistyping the password once, the other session is possibly an attack to guess the password. Thus, the sequence lengths are as important as the patterns.


```python
sgt_darpa = Sgt(kappa=5, lengthsensitive=True, mode='multiprocessing')
```


```python
embedding = sgt_darpa.fit_transform(corpus=sequences)
```


```python
pd.DataFrame(embedding).to_csv(path_or_buf='tmp.csv', index=False)
pd.DataFrame(embedding).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>2391</th>
      <th>2392</th>
      <th>2393</th>
      <th>2394</th>
      <th>2395</th>
      <th>2396</th>
      <th>2397</th>
      <th>2398</th>
      <th>2399</th>
      <th>2400</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.069114</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.804190e-09</td>
      <td>7.041516e-10</td>
      <td>0.0</td>
      <td>2.004958e-12</td>
      <td>0.000132</td>
      <td>1.046458e-07</td>
      <td>5.863092e-16</td>
      <td>7.568986e-23</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.540296</td>
      <td>5.739230e-32</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.785666</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>1.950089e-03</td>
      <td>2.239981e-04</td>
      <td>2.343180e-07</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.528133</td>
      <td>1.576703e-09</td>
      <td>0.0</td>
      <td>2.516644e-29</td>
      <td>1.484843e-57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2401 columns</p>
</div>



#### Applying PCA on the embeddings
The embeddings are sparse. We, therefore, apply PCA on the embeddings.


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=35)
pca.fit(embedding)
X = pca.transform(embedding)
print(np.sum(pca.explained_variance_ratio_))
```

    0.9887812978739061


#### Building a Multi-Layer Perceptron Classifier
The PCA transforms of the embeddings are used directly as inputs to an MLP classifier.


```python
kfold = 3
random_state = 11

test_F1 = np.zeros(kfold)
time_k = np.zeros(kfold)
skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
k = 0
epochs = 300
batch_size = 15

# class_weight = {0 : 1., 1: 1.,}  # The weights can be changed and made inversely proportional to the class size to improve the accuracy.
class_weight = {0 : 0.12, 1: 0.88,}

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],))) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    start_time = time.time()
    model.fit(X_train, y_train ,batch_size=batch_size, epochs=epochs, verbose=1, class_weight=class_weight)
    end_time = time.time()
    time_k[k] = end_time-start_time

    y_pred = model.predict_proba(X_test).round().astype(int)
    y_train_pred = model.predict_proba(X_train).round().astype(int)
    test_F1[k] = sklearn.metrics.f1_score(y_test, y_pred)
    k += 1
```

    Model: "sequential_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_30 (Dense)             (None, 128)               4608      
    _________________________________________________________________
    activation_30 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dropout_20 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_31 (Dense)             (None, 1)                 129       
    _________________________________________________________________
    activation_31 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 4,737
    Trainable params: 4,737
    Non-trainable params: 0
    _________________________________________________________________
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train on 74 samples
    Epoch 1/300
    74/74 [==============================] - 0s 6ms/sample - loss: 0.1404 - accuracy: 0.6216
    Epoch 2/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.1386 - accuracy: 0.6486
    Epoch 3/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.1404 - accuracy: 0.7568
    Epoch 4/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.1309 - accuracy: 0.7297
    Epoch 5/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.1274 - accuracy: 0.7162
    Epoch 6/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.1142 - accuracy: 0.7568
    Epoch 7/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.1041 - accuracy: 0.8784
    Epoch 8/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.1027 - accuracy: 0.8243
    Epoch 9/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0991 - accuracy: 0.8378
    Epoch 10/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0862 - accuracy: 0.8649
    Epoch 11/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0930 - accuracy: 0.8649
    Epoch 12/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0898 - accuracy: 0.8649
    Epoch 13/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0827 - accuracy: 0.8784
    Epoch 14/300
    74/74 [==============================] - 0s 154us/sample - loss: 0.0790 - accuracy: 0.8784
    Epoch 15/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0769 - accuracy: 0.8649
    Epoch 16/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0801 - accuracy: 0.8514
    Epoch 17/300
    74/74 [==============================] - 0s 139us/sample - loss: 0.0740 - accuracy: 0.8784
    Epoch 18/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0723 - accuracy: 0.8649
    Epoch 19/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0679 - accuracy: 0.8649
    Epoch 20/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0704 - accuracy: 0.8919
    Epoch 21/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0621 - accuracy: 0.8649
    Epoch 22/300
    74/74 [==============================] - 0s 133us/sample - loss: 0.0627 - accuracy: 0.8919
    Epoch 23/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0552 - accuracy: 0.8784
    Epoch 24/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0599 - accuracy: 0.8784
    Epoch 25/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0596 - accuracy: 0.8514
    Epoch 26/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0579 - accuracy: 0.8784
    Epoch 27/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0513 - accuracy: 0.8784
    Epoch 28/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0533 - accuracy: 0.8784
    Epoch 29/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0559 - accuracy: 0.8784
    Epoch 30/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0537 - accuracy: 0.8649
    Epoch 31/300
    74/74 [==============================] - 0s 136us/sample - loss: 0.0472 - accuracy: 0.8649
    Epoch 32/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0494 - accuracy: 0.8514
    Epoch 33/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0511 - accuracy: 0.8649
    Epoch 34/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0473 - accuracy: 0.8649
    Epoch 35/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0507 - accuracy: 0.8649
    Epoch 36/300
    74/74 [==============================] - 0s 137us/sample - loss: 0.0468 - accuracy: 0.8649
    Epoch 37/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0459 - accuracy: 0.8649
    Epoch 38/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0428 - accuracy: 0.8649
    Epoch 39/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0439 - accuracy: 0.8649
    Epoch 40/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0388 - accuracy: 0.8649
    Epoch 41/300
    74/74 [==============================] - 0s 133us/sample - loss: 0.0406 - accuracy: 0.8649
    Epoch 42/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0450 - accuracy: 0.8919
    Epoch 43/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0403 - accuracy: 0.8784
    Epoch 44/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0463 - accuracy: 0.8649
    Epoch 45/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0443 - accuracy: 0.8784
    Epoch 46/300
    74/74 [==============================] - 0s 157us/sample - loss: 0.0437 - accuracy: 0.8514
    Epoch 47/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0379 - accuracy: 0.8919
    Epoch 48/300
    74/74 [==============================] - 0s 138us/sample - loss: 0.0388 - accuracy: 0.8784
    Epoch 49/300
    74/74 [==============================] - 0s 142us/sample - loss: 0.0403 - accuracy: 0.8784
    Epoch 50/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0344 - accuracy: 0.8919
    Epoch 51/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0378 - accuracy: 0.8649
    Epoch 52/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0403 - accuracy: 0.8784
    Epoch 53/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0372 - accuracy: 0.9054
    Epoch 54/300
    74/74 [==============================] - 0s 146us/sample - loss: 0.0397 - accuracy: 0.8649
    Epoch 55/300
    74/74 [==============================] - 0s 141us/sample - loss: 0.0408 - accuracy: 0.8784
    Epoch 56/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0422 - accuracy: 0.8649
    Epoch 57/300
    74/74 [==============================] - 0s 143us/sample - loss: 0.0372 - accuracy: 0.8649
    Epoch 58/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0380 - accuracy: 0.8649
    Epoch 59/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0413 - accuracy: 0.8649
    Epoch 60/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0327 - accuracy: 0.8649
    Epoch 61/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0358 - accuracy: 0.8649
    Epoch 62/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0359 - accuracy: 0.8649
    Epoch 63/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0393 - accuracy: 0.8649
    Epoch 64/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0387 - accuracy: 0.8784
    Epoch 65/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0366 - accuracy: 0.8649
    Epoch 66/300
    74/74 [==============================] - 0s 136us/sample - loss: 0.0328 - accuracy: 0.8784
    Epoch 67/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0390 - accuracy: 0.8649
    Epoch 68/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0324 - accuracy: 0.8919
    Epoch 69/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0349 - accuracy: 0.8649
    Epoch 70/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0328 - accuracy: 0.8784
    Epoch 71/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0359 - accuracy: 0.8649
    Epoch 72/300
    74/74 [==============================] - 0s 138us/sample - loss: 0.0383 - accuracy: 0.8514
    Epoch 73/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0366 - accuracy: 0.8649
    Epoch 74/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0359 - accuracy: 0.8919
    Epoch 75/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0395 - accuracy: 0.8514
    Epoch 76/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0363 - accuracy: 0.8649
    Epoch 77/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0346 - accuracy: 0.8784
    Epoch 78/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0370 - accuracy: 0.8649
    Epoch 79/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0319 - accuracy: 0.8919
    Epoch 80/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0349 - accuracy: 0.8649
    Epoch 81/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0365 - accuracy: 0.8649
    Epoch 82/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0359 - accuracy: 0.8514
    Epoch 83/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0319 - accuracy: 0.8784
    Epoch 84/300
    74/74 [==============================] - 0s 138us/sample - loss: 0.0361 - accuracy: 0.8649
    Epoch 85/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0294 - accuracy: 0.8784
    Epoch 86/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0360 - accuracy: 0.8784
    Epoch 87/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0325 - accuracy: 0.8784
    Epoch 88/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0303 - accuracy: 0.8919
    Epoch 89/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0309 - accuracy: 0.8784
    Epoch 90/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0347 - accuracy: 0.8784
    Epoch 91/300
    74/74 [==============================] - 0s 139us/sample - loss: 0.0379 - accuracy: 0.8649
    Epoch 92/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0382 - accuracy: 0.8514
    Epoch 93/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0349 - accuracy: 0.8919
    Epoch 94/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0274 - accuracy: 0.8919
    Epoch 95/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0368 - accuracy: 0.8514
    Epoch 96/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0281 - accuracy: 0.8649
    Epoch 97/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0291 - accuracy: 0.9054
    Epoch 98/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0299 - accuracy: 0.9054
    Epoch 99/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0287 - accuracy: 0.8649
    Epoch 100/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0353 - accuracy: 0.8649
    Epoch 101/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0316 - accuracy: 0.8919
    Epoch 102/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0299 - accuracy: 0.8649
    Epoch 103/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0353 - accuracy: 0.8784
    Epoch 104/300
    74/74 [==============================] - 0s 105us/sample - loss: 0.0347 - accuracy: 0.8514
    Epoch 105/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0294 - accuracy: 0.8784
    Epoch 106/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0344 - accuracy: 0.8784
    Epoch 107/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0323 - accuracy: 0.8919
    Epoch 108/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0297 - accuracy: 0.9189
    Epoch 109/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0333 - accuracy: 0.8649
    Epoch 110/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0300 - accuracy: 0.8649
    Epoch 111/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0369 - accuracy: 0.8514
    Epoch 112/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0323 - accuracy: 0.8919
    Epoch 113/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0361 - accuracy: 0.8919
    Epoch 114/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0336 - accuracy: 0.8649
    Epoch 115/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0291 - accuracy: 0.8649
    Epoch 116/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0351 - accuracy: 0.8649
    Epoch 117/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0288 - accuracy: 0.8649
    Epoch 118/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0329 - accuracy: 0.8919
    Epoch 119/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0393 - accuracy: 0.8784
    Epoch 120/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0234 - accuracy: 0.8919
    Epoch 121/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0381 - accuracy: 0.8784
    Epoch 122/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0319 - accuracy: 0.8784
    Epoch 123/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0286 - accuracy: 0.8919
    Epoch 124/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0335 - accuracy: 0.8784
    Epoch 125/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0324 - accuracy: 0.9054
    Epoch 126/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0268 - accuracy: 0.8784
    Epoch 127/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0359 - accuracy: 0.8649
    Epoch 128/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0326 - accuracy: 0.9054
    Epoch 129/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0305 - accuracy: 0.8784
    Epoch 130/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0306 - accuracy: 0.8784
    Epoch 131/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0318 - accuracy: 0.8649
    Epoch 132/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0312 - accuracy: 0.8784
    Epoch 133/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0330 - accuracy: 0.8919
    Epoch 134/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 135/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0330 - accuracy: 0.8649
    Epoch 136/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0335 - accuracy: 0.8649
    Epoch 137/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0363 - accuracy: 0.8514
    Epoch 138/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0363 - accuracy: 0.8649
    Epoch 139/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0334 - accuracy: 0.8649
    Epoch 140/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0341 - accuracy: 0.8649
    Epoch 141/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0298 - accuracy: 0.8919
    Epoch 142/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0370 - accuracy: 0.8514
    Epoch 143/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0325 - accuracy: 0.8649
    Epoch 144/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0293 - accuracy: 0.8649
    Epoch 145/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0380 - accuracy: 0.8514
    Epoch 146/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0315 - accuracy: 0.8784
    Epoch 147/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0328 - accuracy: 0.8649
    Epoch 148/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 149/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0286 - accuracy: 0.8649
    Epoch 150/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0278 - accuracy: 0.8784
    Epoch 151/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0297 - accuracy: 0.8784
    Epoch 152/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0302 - accuracy: 0.9189
    Epoch 153/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0332 - accuracy: 0.8649
    Epoch 154/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0299 - accuracy: 0.8784
    Epoch 155/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0359 - accuracy: 0.8649
    Epoch 156/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0325 - accuracy: 0.8649
    Epoch 157/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0318 - accuracy: 0.8649
    Epoch 158/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0308 - accuracy: 0.8784
    Epoch 159/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0295 - accuracy: 0.8649
    Epoch 160/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0323 - accuracy: 0.8514
    Epoch 161/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0314 - accuracy: 0.8919
    Epoch 162/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0309 - accuracy: 0.8784
    Epoch 163/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0304 - accuracy: 0.9189
    Epoch 164/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0275 - accuracy: 0.8919
    Epoch 165/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0327 - accuracy: 0.8784
    Epoch 166/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0359 - accuracy: 0.8649
    Epoch 167/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0304 - accuracy: 0.8919
    Epoch 168/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0341 - accuracy: 0.8649
    Epoch 169/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0316 - accuracy: 0.8649
    Epoch 170/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0270 - accuracy: 0.8649
    Epoch 171/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0300 - accuracy: 0.8649
    Epoch 172/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0298 - accuracy: 0.9054
    Epoch 173/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0270 - accuracy: 0.8919
    Epoch 174/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0293 - accuracy: 0.8649
    Epoch 175/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0337 - accuracy: 0.8649
    Epoch 176/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0313 - accuracy: 0.8784
    Epoch 177/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0327 - accuracy: 0.8784
    Epoch 178/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0380 - accuracy: 0.8649
    Epoch 179/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0295 - accuracy: 0.8649
    Epoch 180/300
    74/74 [==============================] - 0s 133us/sample - loss: 0.0337 - accuracy: 0.8514
    Epoch 181/300
    74/74 [==============================] - 0s 137us/sample - loss: 0.0344 - accuracy: 0.8649
    Epoch 182/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0355 - accuracy: 0.8514
    Epoch 183/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0330 - accuracy: 0.8784
    Epoch 184/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0295 - accuracy: 0.8784
    Epoch 185/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0368 - accuracy: 0.8514
    Epoch 186/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0339 - accuracy: 0.8649
    Epoch 187/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0283 - accuracy: 0.8649
    Epoch 188/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0309 - accuracy: 0.8649
    Epoch 189/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0315 - accuracy: 0.8919
    Epoch 190/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0285 - accuracy: 0.8649
    Epoch 191/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0339 - accuracy: 0.8649
    Epoch 192/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0285 - accuracy: 0.8784
    Epoch 193/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0304 - accuracy: 0.8919
    Epoch 194/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0355 - accuracy: 0.8784
    Epoch 195/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0392 - accuracy: 0.8514
    Epoch 196/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0282 - accuracy: 0.8784
    Epoch 197/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0302 - accuracy: 0.8649
    Epoch 198/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0324 - accuracy: 0.8649
    Epoch 199/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0274 - accuracy: 0.8784
    Epoch 200/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0289 - accuracy: 0.8784
    Epoch 201/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0375 - accuracy: 0.8514
    Epoch 202/300
    74/74 [==============================] - 0s 133us/sample - loss: 0.0337 - accuracy: 0.8649
    Epoch 203/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0329 - accuracy: 0.8649
    Epoch 204/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0303 - accuracy: 0.8649
    Epoch 205/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0335 - accuracy: 0.8784
    Epoch 206/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0304 - accuracy: 0.8649
    Epoch 207/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0339 - accuracy: 0.8649
    Epoch 208/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0261 - accuracy: 0.8784
    Epoch 209/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0304 - accuracy: 0.8649
    Epoch 210/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0303 - accuracy: 0.8649
    Epoch 211/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0318 - accuracy: 0.8784
    Epoch 212/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0358 - accuracy: 0.8919
    Epoch 213/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0272 - accuracy: 0.8784
    Epoch 214/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0293 - accuracy: 0.8649
    Epoch 215/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0347 - accuracy: 0.8649
    Epoch 216/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0302 - accuracy: 0.8649
    Epoch 217/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0331 - accuracy: 0.8784
    Epoch 218/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0283 - accuracy: 0.8784
    Epoch 219/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0329 - accuracy: 0.8649
    Epoch 220/300
    74/74 [==============================] - 0s 136us/sample - loss: 0.0291 - accuracy: 0.8919
    Epoch 221/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0323 - accuracy: 0.8784
    Epoch 222/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0341 - accuracy: 0.8784
    Epoch 223/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0310 - accuracy: 0.8919
    Epoch 224/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0337 - accuracy: 0.8784
    Epoch 225/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0359 - accuracy: 0.8649
    Epoch 226/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0355 - accuracy: 0.8649
    Epoch 227/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 228/300
    74/74 [==============================] - 0s 140us/sample - loss: 0.0353 - accuracy: 0.8649
    Epoch 229/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0323 - accuracy: 0.8784
    Epoch 230/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0332 - accuracy: 0.8649
    Epoch 231/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0350 - accuracy: 0.8649
    Epoch 232/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0279 - accuracy: 0.8919
    Epoch 233/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 234/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0334 - accuracy: 0.8649
    Epoch 235/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0327 - accuracy: 0.8649
    Epoch 236/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0316 - accuracy: 0.8649
    Epoch 237/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0292 - accuracy: 0.8919
    Epoch 238/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0320 - accuracy: 0.8919
    Epoch 239/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0312 - accuracy: 0.8649
    Epoch 240/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0332 - accuracy: 0.8649
    Epoch 241/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0296 - accuracy: 0.8649
    Epoch 242/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0331 - accuracy: 0.8649
    Epoch 243/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0258 - accuracy: 0.8784
    Epoch 244/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0316 - accuracy: 0.8919
    Epoch 245/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0321 - accuracy: 0.8784
    Epoch 246/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0306 - accuracy: 0.8649
    Epoch 247/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0319 - accuracy: 0.8649
    Epoch 248/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0275 - accuracy: 0.8784
    Epoch 249/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0318 - accuracy: 0.8649
    Epoch 250/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0324 - accuracy: 0.8649
    Epoch 251/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0311 - accuracy: 0.8919
    Epoch 252/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0335 - accuracy: 0.8649
    Epoch 253/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0334 - accuracy: 0.8649
    Epoch 254/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0359 - accuracy: 0.8514
    Epoch 255/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0326 - accuracy: 0.8784
    Epoch 256/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 257/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0343 - accuracy: 0.8784
    Epoch 258/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0309 - accuracy: 0.8649
    Epoch 259/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0301 - accuracy: 0.8649
    Epoch 260/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0315 - accuracy: 0.8649
    Epoch 261/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0342 - accuracy: 0.8649
    Epoch 262/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0300 - accuracy: 0.8649
    Epoch 263/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0358 - accuracy: 0.8649
    Epoch 264/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0295 - accuracy: 0.8649
    Epoch 265/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0356 - accuracy: 0.8649
    Epoch 266/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0318 - accuracy: 0.8649
    Epoch 267/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0298 - accuracy: 0.8784
    Epoch 268/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0278 - accuracy: 0.8649
    Epoch 269/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0302 - accuracy: 0.8649
    Epoch 270/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0305 - accuracy: 0.8649
    Epoch 271/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0324 - accuracy: 0.8649
    Epoch 272/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0322 - accuracy: 0.8784
    Epoch 273/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0302 - accuracy: 0.8649
    Epoch 274/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0309 - accuracy: 0.8649
    Epoch 275/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0296 - accuracy: 0.8649
    Epoch 276/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0285 - accuracy: 0.8649
    Epoch 277/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0324 - accuracy: 0.8649
    Epoch 278/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0349 - accuracy: 0.8514
    Epoch 279/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0347 - accuracy: 0.8649
    Epoch 280/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0320 - accuracy: 0.8649
    Epoch 281/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0350 - accuracy: 0.8784
    Epoch 282/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0320 - accuracy: 0.8649
    Epoch 283/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0331 - accuracy: 0.8649
    Epoch 284/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0283 - accuracy: 0.8649
    Epoch 285/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 286/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0306 - accuracy: 0.8649
    Epoch 287/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0306 - accuracy: 0.8784
    Epoch 288/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0324 - accuracy: 0.8649
    Epoch 289/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0347 - accuracy: 0.8514
    Epoch 290/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0362 - accuracy: 0.8514
    Epoch 291/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0330 - accuracy: 0.8649
    Epoch 292/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0306 - accuracy: 0.8649
    Epoch 293/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0326 - accuracy: 0.8649
    Epoch 294/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0346 - accuracy: 0.8649
    Epoch 295/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0335 - accuracy: 0.8649
    Epoch 296/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0304 - accuracy: 0.8649
    Epoch 297/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0303 - accuracy: 0.8784
    Epoch 298/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0324 - accuracy: 0.8649
    Epoch 299/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 300/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0297 - accuracy: 0.8649
    Model: "sequential_11"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_32 (Dense)             (None, 128)               4608      
    _________________________________________________________________
    activation_32 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dropout_21 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_33 (Dense)             (None, 1)                 129       
    _________________________________________________________________
    activation_33 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 4,737
    Trainable params: 4,737
    Non-trainable params: 0
    _________________________________________________________________
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train on 74 samples
    Epoch 1/300
    74/74 [==============================] - 0s 6ms/sample - loss: 0.1394 - accuracy: 0.6757
    Epoch 2/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.1322 - accuracy: 0.7568
    Epoch 3/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.1254 - accuracy: 0.7973
    Epoch 4/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.1130 - accuracy: 0.7973
    Epoch 5/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.1276 - accuracy: 0.7568
    Epoch 6/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.1141 - accuracy: 0.9054
    Epoch 7/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.1047 - accuracy: 0.8514
    Epoch 8/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.1044 - accuracy: 0.8784
    Epoch 9/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.1066 - accuracy: 0.8919
    Epoch 10/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0914 - accuracy: 0.8919
    Epoch 11/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0893 - accuracy: 0.9054
    Epoch 12/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0854 - accuracy: 0.9054
    Epoch 13/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0835 - accuracy: 0.8919
    Epoch 14/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0761 - accuracy: 0.9054
    Epoch 15/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0776 - accuracy: 0.9189
    Epoch 16/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0744 - accuracy: 0.9189
    Epoch 17/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0717 - accuracy: 0.9189
    Epoch 18/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0722 - accuracy: 0.9054
    Epoch 19/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0662 - accuracy: 0.8919
    Epoch 20/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0679 - accuracy: 0.9189
    Epoch 21/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0633 - accuracy: 0.9189
    Epoch 22/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0597 - accuracy: 0.9189
    Epoch 23/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0615 - accuracy: 0.8919
    Epoch 24/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0586 - accuracy: 0.9189
    Epoch 25/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0625 - accuracy: 0.9054
    Epoch 26/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0528 - accuracy: 0.9189
    Epoch 27/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0592 - accuracy: 0.9054
    Epoch 28/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0547 - accuracy: 0.9189
    Epoch 29/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0532 - accuracy: 0.9054
    Epoch 30/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0529 - accuracy: 0.9189
    Epoch 31/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0491 - accuracy: 0.9189
    Epoch 32/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0557 - accuracy: 0.9189
    Epoch 33/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0480 - accuracy: 0.9189
    Epoch 34/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0480 - accuracy: 0.9189
    Epoch 35/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0449 - accuracy: 0.9189
    Epoch 36/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0452 - accuracy: 0.9189
    Epoch 37/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0472 - accuracy: 0.9189
    Epoch 38/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0494 - accuracy: 0.9189
    Epoch 39/300
    74/74 [==============================] - 0s 171us/sample - loss: 0.0431 - accuracy: 0.9189
    Epoch 40/300
    74/74 [==============================] - 0s 157us/sample - loss: 0.0429 - accuracy: 0.9189
    Epoch 41/300
    74/74 [==============================] - 0s 151us/sample - loss: 0.0438 - accuracy: 0.9189
    Epoch 42/300
    74/74 [==============================] - 0s 177us/sample - loss: 0.0412 - accuracy: 0.9189
    Epoch 43/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0427 - accuracy: 0.9189
    Epoch 44/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0416 - accuracy: 0.9054
    Epoch 45/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0382 - accuracy: 0.9189
    Epoch 46/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0392 - accuracy: 0.9189
    Epoch 47/300
    74/74 [==============================] - 0s 106us/sample - loss: 0.0406 - accuracy: 0.9189
    Epoch 48/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0368 - accuracy: 0.9189
    Epoch 49/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0412 - accuracy: 0.9189
    Epoch 50/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0352 - accuracy: 0.9189
    Epoch 51/300
    74/74 [==============================] - 0s 138us/sample - loss: 0.0420 - accuracy: 0.9189
    Epoch 52/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0377 - accuracy: 0.9189
    Epoch 53/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0392 - accuracy: 0.9189
    Epoch 54/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0380 - accuracy: 0.9189
    Epoch 55/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0352 - accuracy: 0.9189
    Epoch 56/300
    74/74 [==============================] - 0s 140us/sample - loss: 0.0348 - accuracy: 0.9189
    Epoch 57/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0365 - accuracy: 0.9189
    Epoch 58/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0319 - accuracy: 0.9189
    Epoch 59/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0350 - accuracy: 0.9054
    Epoch 60/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0373 - accuracy: 0.9189
    Epoch 61/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0382 - accuracy: 0.9189
    Epoch 62/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0346 - accuracy: 0.9189
    Epoch 63/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0312 - accuracy: 0.9189
    Epoch 64/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0349 - accuracy: 0.9189
    Epoch 65/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0345 - accuracy: 0.9189
    Epoch 66/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0312 - accuracy: 0.9189
    Epoch 67/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0322 - accuracy: 0.9189
    Epoch 68/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0307 - accuracy: 0.9189
    Epoch 69/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0308 - accuracy: 0.9189
    Epoch 70/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0368 - accuracy: 0.9189
    Epoch 71/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0312 - accuracy: 0.9189
    Epoch 72/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0337 - accuracy: 0.9189
    Epoch 73/300
    74/74 [==============================] - 0s 136us/sample - loss: 0.0319 - accuracy: 0.9189
    Epoch 74/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0344 - accuracy: 0.9189
    Epoch 75/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0338 - accuracy: 0.9189
    Epoch 76/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0303 - accuracy: 0.9189
    Epoch 77/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0303 - accuracy: 0.9189
    Epoch 78/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0250 - accuracy: 0.9189
    Epoch 79/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0301 - accuracy: 0.9189
    Epoch 80/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0333 - accuracy: 0.9189
    Epoch 81/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0294 - accuracy: 0.9189
    Epoch 82/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0271 - accuracy: 0.9189
    Epoch 83/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0327 - accuracy: 0.9189
    Epoch 84/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0297 - accuracy: 0.9189
    Epoch 85/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0309 - accuracy: 0.9189
    Epoch 86/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0331 - accuracy: 0.9189
    Epoch 87/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0327 - accuracy: 0.9189
    Epoch 88/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0323 - accuracy: 0.9189
    Epoch 89/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0275 - accuracy: 0.9189
    Epoch 90/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0304 - accuracy: 0.9189
    Epoch 91/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0293 - accuracy: 0.9189
    Epoch 92/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0331 - accuracy: 0.9189
    Epoch 93/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0298 - accuracy: 0.9189
    Epoch 94/300
    74/74 [==============================] - 0s 136us/sample - loss: 0.0311 - accuracy: 0.9189
    Epoch 95/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0310 - accuracy: 0.9189
    Epoch 96/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0309 - accuracy: 0.9189
    Epoch 97/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0292 - accuracy: 0.9189
    Epoch 98/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0311 - accuracy: 0.9189
    Epoch 99/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0261 - accuracy: 0.9189
    Epoch 100/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0305 - accuracy: 0.9189
    Epoch 101/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0282 - accuracy: 0.9189
    Epoch 102/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0355 - accuracy: 0.9189
    Epoch 103/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0281 - accuracy: 0.9189
    Epoch 104/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0286 - accuracy: 0.9189
    Epoch 105/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0310 - accuracy: 0.9189
    Epoch 106/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0269 - accuracy: 0.9189
    Epoch 107/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0344 - accuracy: 0.9189
    Epoch 108/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0323 - accuracy: 0.9189
    Epoch 109/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0309 - accuracy: 0.9189
    Epoch 110/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0258 - accuracy: 0.9189
    Epoch 111/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0305 - accuracy: 0.9189
    Epoch 112/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0274 - accuracy: 0.9189
    Epoch 113/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0268 - accuracy: 0.9189
    Epoch 114/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0283 - accuracy: 0.9189
    Epoch 115/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0274 - accuracy: 0.9189
    Epoch 116/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0311 - accuracy: 0.9189
    Epoch 117/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0318 - accuracy: 0.9189
    Epoch 118/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0265 - accuracy: 0.9189
    Epoch 119/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0305 - accuracy: 0.9189
    Epoch 120/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0258 - accuracy: 0.9189
    Epoch 121/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0269 - accuracy: 0.9189
    Epoch 122/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0323 - accuracy: 0.9189
    Epoch 123/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0291 - accuracy: 0.9189
    Epoch 124/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0292 - accuracy: 0.9189
    Epoch 125/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0258 - accuracy: 0.9189
    Epoch 126/300
    74/74 [==============================] - 0s 103us/sample - loss: 0.0257 - accuracy: 0.9189
    Epoch 127/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0321 - accuracy: 0.9189
    Epoch 128/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0274 - accuracy: 0.9189
    Epoch 129/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0284 - accuracy: 0.9189
    Epoch 130/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0285 - accuracy: 0.9189
    Epoch 131/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0272 - accuracy: 0.9189
    Epoch 132/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0295 - accuracy: 0.9189
    Epoch 133/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0273 - accuracy: 0.9189
    Epoch 134/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0293 - accuracy: 0.9189
    Epoch 135/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0295 - accuracy: 0.9189
    Epoch 136/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0271 - accuracy: 0.9189
    Epoch 137/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0292 - accuracy: 0.9189
    Epoch 138/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0285 - accuracy: 0.9189
    Epoch 139/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0299 - accuracy: 0.9189
    Epoch 140/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0308 - accuracy: 0.9189
    Epoch 141/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0256 - accuracy: 0.9189
    Epoch 142/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0259 - accuracy: 0.9189
    Epoch 143/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0256 - accuracy: 0.9189
    Epoch 144/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0294 - accuracy: 0.9189
    Epoch 145/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0275 - accuracy: 0.9189
    Epoch 146/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0321 - accuracy: 0.9189
    Epoch 147/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0258 - accuracy: 0.9189
    Epoch 148/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0271 - accuracy: 0.9189
    Epoch 149/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0286 - accuracy: 0.9189
    Epoch 150/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0337 - accuracy: 0.9189
    Epoch 151/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0277 - accuracy: 0.9189
    Epoch 152/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0290 - accuracy: 0.9189
    Epoch 153/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0259 - accuracy: 0.9189
    Epoch 154/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0257 - accuracy: 0.9189
    Epoch 155/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0263 - accuracy: 0.9189
    Epoch 156/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0299 - accuracy: 0.9189
    Epoch 157/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0283 - accuracy: 0.9189
    Epoch 158/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0325 - accuracy: 0.9189
    Epoch 159/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0247 - accuracy: 0.9189
    Epoch 160/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0305 - accuracy: 0.9189
    Epoch 161/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0252 - accuracy: 0.9189
    Epoch 162/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0282 - accuracy: 0.9189
    Epoch 163/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0255 - accuracy: 0.9189
    Epoch 164/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0265 - accuracy: 0.9189
    Epoch 165/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0250 - accuracy: 0.9189
    Epoch 166/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0300 - accuracy: 0.9189
    Epoch 167/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0283 - accuracy: 0.9189
    Epoch 168/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0321 - accuracy: 0.9189
    Epoch 169/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0246 - accuracy: 0.9189
    Epoch 170/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0230 - accuracy: 0.9189
    Epoch 171/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0268 - accuracy: 0.9189
    Epoch 172/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0263 - accuracy: 0.9189
    Epoch 173/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0305 - accuracy: 0.9189
    Epoch 174/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0262 - accuracy: 0.9189
    Epoch 175/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0276 - accuracy: 0.9189
    Epoch 176/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0254 - accuracy: 0.9189
    Epoch 177/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0311 - accuracy: 0.9189
    Epoch 178/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0311 - accuracy: 0.9189
    Epoch 179/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0278 - accuracy: 0.9189
    Epoch 180/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0313 - accuracy: 0.9189
    Epoch 181/300
    74/74 [==============================] - 0s 133us/sample - loss: 0.0291 - accuracy: 0.9189
    Epoch 182/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0277 - accuracy: 0.9189
    Epoch 183/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0276 - accuracy: 0.9189
    Epoch 184/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0247 - accuracy: 0.9189
    Epoch 185/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0314 - accuracy: 0.9189
    Epoch 186/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0270 - accuracy: 0.9189
    Epoch 187/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0306 - accuracy: 0.9189
    Epoch 188/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0250 - accuracy: 0.9189
    Epoch 189/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0280 - accuracy: 0.9189
    Epoch 190/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0304 - accuracy: 0.9189
    Epoch 191/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0286 - accuracy: 0.9189
    Epoch 192/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0278 - accuracy: 0.9189
    Epoch 193/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0225 - accuracy: 0.9189
    Epoch 194/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0266 - accuracy: 0.9189
    Epoch 195/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0260 - accuracy: 0.9189
    Epoch 196/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0254 - accuracy: 0.9189
    Epoch 197/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0268 - accuracy: 0.9189
    Epoch 198/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0248 - accuracy: 0.9189
    Epoch 199/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0285 - accuracy: 0.9189
    Epoch 200/300
    74/74 [==============================] - 0s 140us/sample - loss: 0.0237 - accuracy: 0.9189
    Epoch 201/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0291 - accuracy: 0.9189
    Epoch 202/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0290 - accuracy: 0.9189
    Epoch 203/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0292 - accuracy: 0.9189
    Epoch 204/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0307 - accuracy: 0.9189
    Epoch 205/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0264 - accuracy: 0.9189
    Epoch 206/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0288 - accuracy: 0.9189
    Epoch 207/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0305 - accuracy: 0.9189
    Epoch 208/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0252 - accuracy: 0.9189
    Epoch 209/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0273 - accuracy: 0.9189
    Epoch 210/300
    74/74 [==============================] - 0s 138us/sample - loss: 0.0268 - accuracy: 0.9189
    Epoch 211/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0259 - accuracy: 0.9189
    Epoch 212/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0253 - accuracy: 0.9189
    Epoch 213/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0244 - accuracy: 0.9189
    Epoch 214/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0276 - accuracy: 0.9189
    Epoch 215/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0281 - accuracy: 0.9189
    Epoch 216/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0260 - accuracy: 0.9189
    Epoch 217/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0265 - accuracy: 0.9189
    Epoch 218/300
    74/74 [==============================] - 0s 135us/sample - loss: 0.0301 - accuracy: 0.9189
    Epoch 219/300
    74/74 [==============================] - 0s 143us/sample - loss: 0.0279 - accuracy: 0.9189
    Epoch 220/300
    74/74 [==============================] - 0s 144us/sample - loss: 0.0254 - accuracy: 0.9189
    Epoch 221/300
    74/74 [==============================] - 0s 140us/sample - loss: 0.0259 - accuracy: 0.9189
    Epoch 222/300
    74/74 [==============================] - 0s 140us/sample - loss: 0.0238 - accuracy: 0.9189
    Epoch 223/300
    74/74 [==============================] - 0s 135us/sample - loss: 0.0303 - accuracy: 0.9189
    Epoch 224/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0244 - accuracy: 0.9189
    Epoch 225/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0270 - accuracy: 0.9189
    Epoch 226/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0297 - accuracy: 0.9189
    Epoch 227/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0266 - accuracy: 0.9189
    Epoch 228/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0265 - accuracy: 0.9189
    Epoch 229/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0240 - accuracy: 0.9189
    Epoch 230/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0268 - accuracy: 0.9189
    Epoch 231/300
    74/74 [==============================] - 0s 141us/sample - loss: 0.0309 - accuracy: 0.9189
    Epoch 232/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0305 - accuracy: 0.9189
    Epoch 233/300
    74/74 [==============================] - 0s 143us/sample - loss: 0.0279 - accuracy: 0.9189
    Epoch 234/300
    74/74 [==============================] - 0s 149us/sample - loss: 0.0267 - accuracy: 0.9189
    Epoch 235/300
    74/74 [==============================] - 0s 140us/sample - loss: 0.0281 - accuracy: 0.9189
    Epoch 236/300
    74/74 [==============================] - 0s 145us/sample - loss: 0.0283 - accuracy: 0.9189
    Epoch 237/300
    74/74 [==============================] - 0s 160us/sample - loss: 0.0263 - accuracy: 0.9189
    Epoch 238/300
    74/74 [==============================] - 0s 135us/sample - loss: 0.0286 - accuracy: 0.9189
    Epoch 239/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0279 - accuracy: 0.9189
    Epoch 240/300
    74/74 [==============================] - 0s 142us/sample - loss: 0.0254 - accuracy: 0.9189
    Epoch 241/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0297 - accuracy: 0.9189
    Epoch 242/300
    74/74 [==============================] - 0s 146us/sample - loss: 0.0276 - accuracy: 0.9189
    Epoch 243/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0280 - accuracy: 0.9189
    Epoch 244/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0280 - accuracy: 0.9189
    Epoch 245/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0295 - accuracy: 0.9189
    Epoch 246/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0250 - accuracy: 0.9189
    Epoch 247/300
    74/74 [==============================] - 0s 133us/sample - loss: 0.0298 - accuracy: 0.9189
    Epoch 248/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0273 - accuracy: 0.9189
    Epoch 249/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0283 - accuracy: 0.9189
    Epoch 250/300
    74/74 [==============================] - 0s 136us/sample - loss: 0.0232 - accuracy: 0.9189
    Epoch 251/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0288 - accuracy: 0.9189
    Epoch 252/300
    74/74 [==============================] - 0s 135us/sample - loss: 0.0255 - accuracy: 0.9189
    Epoch 253/300
    74/74 [==============================] - 0s 146us/sample - loss: 0.0268 - accuracy: 0.9189
    Epoch 254/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0249 - accuracy: 0.9189
    Epoch 255/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0269 - accuracy: 0.9189
    Epoch 256/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0259 - accuracy: 0.9189
    Epoch 257/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0259 - accuracy: 0.9189
    Epoch 258/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0247 - accuracy: 0.9189
    Epoch 259/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0288 - accuracy: 0.9189
    Epoch 260/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0272 - accuracy: 0.9189
    Epoch 261/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0262 - accuracy: 0.9189
    Epoch 262/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0285 - accuracy: 0.9189
    Epoch 263/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0269 - accuracy: 0.9189
    Epoch 264/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0289 - accuracy: 0.9189
    Epoch 265/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0248 - accuracy: 0.9189
    Epoch 266/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0243 - accuracy: 0.9189
    Epoch 267/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0282 - accuracy: 0.9189
    Epoch 268/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0269 - accuracy: 0.9189
    Epoch 269/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0268 - accuracy: 0.9189
    Epoch 270/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0272 - accuracy: 0.9189
    Epoch 271/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0258 - accuracy: 0.9189
    Epoch 272/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0255 - accuracy: 0.9189
    Epoch 273/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0279 - accuracy: 0.9189
    Epoch 274/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0262 - accuracy: 0.9189
    Epoch 275/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0281 - accuracy: 0.9189
    Epoch 276/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0247 - accuracy: 0.9189
    Epoch 277/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0245 - accuracy: 0.9189
    Epoch 278/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0279 - accuracy: 0.9189
    Epoch 279/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0309 - accuracy: 0.9189
    Epoch 280/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0240 - accuracy: 0.9189
    Epoch 281/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0265 - accuracy: 0.9189
    Epoch 282/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0267 - accuracy: 0.9189
    Epoch 283/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0292 - accuracy: 0.9189
    Epoch 284/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0270 - accuracy: 0.9189
    Epoch 285/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0267 - accuracy: 0.9189
    Epoch 286/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0299 - accuracy: 0.9189
    Epoch 287/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0283 - accuracy: 0.9189
    Epoch 288/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0248 - accuracy: 0.9189
    Epoch 289/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0257 - accuracy: 0.9189
    Epoch 290/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0257 - accuracy: 0.9189
    Epoch 291/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0272 - accuracy: 0.9189
    Epoch 292/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0301 - accuracy: 0.9189
    Epoch 293/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0321 - accuracy: 0.9189
    Epoch 294/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0241 - accuracy: 0.9189
    Epoch 295/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0277 - accuracy: 0.9189
    Epoch 296/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0254 - accuracy: 0.9189
    Epoch 297/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0263 - accuracy: 0.9189
    Epoch 298/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0276 - accuracy: 0.9189
    Epoch 299/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0225 - accuracy: 0.9189
    Epoch 300/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0309 - accuracy: 0.9189
    Model: "sequential_12"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_34 (Dense)             (None, 128)               4608      
    _________________________________________________________________
    activation_34 (Activation)   (None, 128)               0         
    _________________________________________________________________
    dropout_22 (Dropout)         (None, 128)               0         
    _________________________________________________________________
    dense_35 (Dense)             (None, 1)                 129       
    _________________________________________________________________
    activation_35 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 4,737
    Trainable params: 4,737
    Non-trainable params: 0
    _________________________________________________________________
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train on 74 samples
    Epoch 1/300
    74/74 [==============================] - 0s 6ms/sample - loss: 0.1366 - accuracy: 0.6081
    Epoch 2/300
    74/74 [==============================] - 0s 142us/sample - loss: 0.1310 - accuracy: 0.7027
    Epoch 3/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.1200 - accuracy: 0.7027
    Epoch 4/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.1165 - accuracy: 0.7568
    Epoch 5/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.1130 - accuracy: 0.7973
    Epoch 6/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.1052 - accuracy: 0.7973
    Epoch 7/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.1005 - accuracy: 0.8649
    Epoch 8/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0983 - accuracy: 0.7973
    Epoch 9/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0957 - accuracy: 0.7838
    Epoch 10/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0917 - accuracy: 0.8514
    Epoch 11/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0894 - accuracy: 0.8243
    Epoch 12/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0860 - accuracy: 0.8514
    Epoch 13/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0821 - accuracy: 0.8243
    Epoch 14/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0786 - accuracy: 0.8378
    Epoch 15/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0756 - accuracy: 0.8514
    Epoch 16/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0779 - accuracy: 0.8784
    Epoch 17/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0761 - accuracy: 0.8514
    Epoch 18/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0713 - accuracy: 0.8378
    Epoch 19/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0686 - accuracy: 0.8649
    Epoch 20/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0678 - accuracy: 0.8649
    Epoch 21/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0627 - accuracy: 0.8784
    Epoch 22/300
    74/74 [==============================] - 0s 135us/sample - loss: 0.0680 - accuracy: 0.8649
    Epoch 23/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0629 - accuracy: 0.8514
    Epoch 24/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0601 - accuracy: 0.8649
    Epoch 25/300
    74/74 [==============================] - 0s 132us/sample - loss: 0.0617 - accuracy: 0.8514
    Epoch 26/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0586 - accuracy: 0.8649
    Epoch 27/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0536 - accuracy: 0.8784
    Epoch 28/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0583 - accuracy: 0.8649
    Epoch 29/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0503 - accuracy: 0.8784
    Epoch 30/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0528 - accuracy: 0.8784
    Epoch 31/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0542 - accuracy: 0.8514
    Epoch 32/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0519 - accuracy: 0.8649
    Epoch 33/300
    74/74 [==============================] - 0s 105us/sample - loss: 0.0576 - accuracy: 0.8514
    Epoch 34/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0538 - accuracy: 0.8649
    Epoch 35/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0475 - accuracy: 0.8649
    Epoch 36/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0530 - accuracy: 0.8514
    Epoch 37/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0467 - accuracy: 0.8784
    Epoch 38/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0480 - accuracy: 0.8649
    Epoch 39/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0444 - accuracy: 0.8784
    Epoch 40/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0432 - accuracy: 0.8784
    Epoch 41/300
    74/74 [==============================] - 0s 138us/sample - loss: 0.0445 - accuracy: 0.8649
    Epoch 42/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0530 - accuracy: 0.8514
    Epoch 43/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0483 - accuracy: 0.8649
    Epoch 44/300
    74/74 [==============================] - 0s 151us/sample - loss: 0.0430 - accuracy: 0.8919
    Epoch 45/300
    74/74 [==============================] - 0s 159us/sample - loss: 0.0415 - accuracy: 0.8649
    Epoch 46/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0478 - accuracy: 0.8514
    Epoch 47/300
    74/74 [==============================] - 0s 138us/sample - loss: 0.0407 - accuracy: 0.8649
    Epoch 48/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0394 - accuracy: 0.8649
    Epoch 49/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0384 - accuracy: 0.8649
    Epoch 50/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0406 - accuracy: 0.8649
    Epoch 51/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0440 - accuracy: 0.8649
    Epoch 52/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0469 - accuracy: 0.8649
    Epoch 53/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0407 - accuracy: 0.8649
    Epoch 54/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0403 - accuracy: 0.8784
    Epoch 55/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0400 - accuracy: 0.8649
    Epoch 56/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0371 - accuracy: 0.8784
    Epoch 57/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0431 - accuracy: 0.8784
    Epoch 58/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0351 - accuracy: 0.8649
    Epoch 59/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0371 - accuracy: 0.8649
    Epoch 60/300
    74/74 [==============================] - 0s 136us/sample - loss: 0.0380 - accuracy: 0.8784
    Epoch 61/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0354 - accuracy: 0.8919
    Epoch 62/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0371 - accuracy: 0.8649
    Epoch 63/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0360 - accuracy: 0.8784
    Epoch 64/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0401 - accuracy: 0.8649
    Epoch 65/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0412 - accuracy: 0.8649
    Epoch 66/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0402 - accuracy: 0.8649
    Epoch 67/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0416 - accuracy: 0.8649
    Epoch 68/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0368 - accuracy: 0.8649
    Epoch 69/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0395 - accuracy: 0.8649
    Epoch 70/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0357 - accuracy: 0.8649
    Epoch 71/300
    74/74 [==============================] - 0s 135us/sample - loss: 0.0365 - accuracy: 0.8649
    Epoch 72/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0360 - accuracy: 0.8784
    Epoch 73/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0360 - accuracy: 0.8649
    Epoch 74/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0324 - accuracy: 0.8649
    Epoch 75/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0328 - accuracy: 0.8649
    Epoch 76/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0390 - accuracy: 0.8784
    Epoch 77/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0297 - accuracy: 0.8919
    Epoch 78/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0415 - accuracy: 0.8649
    Epoch 79/300
    74/74 [==============================] - 0s 142us/sample - loss: 0.0354 - accuracy: 0.8784
    Epoch 80/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0345 - accuracy: 0.8784
    Epoch 81/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0377 - accuracy: 0.8784
    Epoch 82/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0308 - accuracy: 0.8649
    Epoch 83/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0358 - accuracy: 0.8784
    Epoch 84/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0334 - accuracy: 0.8649
    Epoch 85/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0348 - accuracy: 0.8514
    Epoch 86/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0329 - accuracy: 0.8784
    Epoch 87/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0290 - accuracy: 0.8919
    Epoch 88/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 89/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0373 - accuracy: 0.8514
    Epoch 90/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0381 - accuracy: 0.8649
    Epoch 91/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0346 - accuracy: 0.8649
    Epoch 92/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0322 - accuracy: 0.8649
    Epoch 93/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0305 - accuracy: 0.8919
    Epoch 94/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0336 - accuracy: 0.8649
    Epoch 95/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0360 - accuracy: 0.8784
    Epoch 96/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0322 - accuracy: 0.8784
    Epoch 97/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0354 - accuracy: 0.8784
    Epoch 98/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0312 - accuracy: 0.8919
    Epoch 99/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0387 - accuracy: 0.8649
    Epoch 100/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0311 - accuracy: 0.8784
    Epoch 101/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0271 - accuracy: 0.8919
    Epoch 102/300
    74/74 [==============================] - 0s 130us/sample - loss: 0.0351 - accuracy: 0.8784
    Epoch 103/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0384 - accuracy: 0.8919
    Epoch 104/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0349 - accuracy: 0.8649
    Epoch 105/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0350 - accuracy: 0.8649
    Epoch 106/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0337 - accuracy: 0.9054
    Epoch 107/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0355 - accuracy: 0.8649
    Epoch 108/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0310 - accuracy: 0.8784
    Epoch 109/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0339 - accuracy: 0.8649
    Epoch 110/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0336 - accuracy: 0.8784
    Epoch 111/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0296 - accuracy: 0.8649
    Epoch 112/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0323 - accuracy: 0.8919
    Epoch 113/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0323 - accuracy: 0.8784
    Epoch 114/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0350 - accuracy: 0.8649
    Epoch 115/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0325 - accuracy: 0.8649
    Epoch 116/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 117/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0316 - accuracy: 0.8784
    Epoch 118/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0305 - accuracy: 0.9054
    Epoch 119/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0377 - accuracy: 0.8784
    Epoch 120/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0328 - accuracy: 0.8919
    Epoch 121/300
    74/74 [==============================] - 0s 134us/sample - loss: 0.0345 - accuracy: 0.8649
    Epoch 122/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0392 - accuracy: 0.8649
    Epoch 123/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0340 - accuracy: 0.8784
    Epoch 124/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0294 - accuracy: 0.8919
    Epoch 125/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0351 - accuracy: 0.8649
    Epoch 126/300
    74/74 [==============================] - 0s 128us/sample - loss: 0.0322 - accuracy: 0.8649
    Epoch 127/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0325 - accuracy: 0.8649
    Epoch 128/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0371 - accuracy: 0.8514
    Epoch 129/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0303 - accuracy: 0.8784
    Epoch 130/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0398 - accuracy: 0.8514
    Epoch 131/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0323 - accuracy: 0.8784
    Epoch 132/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0292 - accuracy: 0.8784
    Epoch 133/300
    74/74 [==============================] - 0s 123us/sample - loss: 0.0293 - accuracy: 0.9054
    Epoch 134/300
    74/74 [==============================] - 0s 106us/sample - loss: 0.0304 - accuracy: 0.8784
    Epoch 135/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0292 - accuracy: 0.8919
    Epoch 136/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0385 - accuracy: 0.8514
    Epoch 137/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0302 - accuracy: 0.8784
    Epoch 138/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0291 - accuracy: 0.8784
    Epoch 139/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0323 - accuracy: 0.8919
    Epoch 140/300
    74/74 [==============================] - 0s 142us/sample - loss: 0.0307 - accuracy: 0.8784
    Epoch 141/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0298 - accuracy: 0.8784
    Epoch 142/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0295 - accuracy: 0.8784
    Epoch 143/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0334 - accuracy: 0.8649
    Epoch 144/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0319 - accuracy: 0.8649
    Epoch 145/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0347 - accuracy: 0.8649
    Epoch 146/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0310 - accuracy: 0.8649
    Epoch 147/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0340 - accuracy: 0.8649
    Epoch 148/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0349 - accuracy: 0.8784
    Epoch 149/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0291 - accuracy: 0.8784
    Epoch 150/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0288 - accuracy: 0.8649
    Epoch 151/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0370 - accuracy: 0.8784
    Epoch 152/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0298 - accuracy: 0.8784
    Epoch 153/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0328 - accuracy: 0.8784
    Epoch 154/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0327 - accuracy: 0.8649
    Epoch 155/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0340 - accuracy: 0.8514
    Epoch 156/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0276 - accuracy: 0.8649
    Epoch 157/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0308 - accuracy: 0.8649
    Epoch 158/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0373 - accuracy: 0.8649
    Epoch 159/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0297 - accuracy: 0.8649
    Epoch 160/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0347 - accuracy: 0.8784
    Epoch 161/300
    74/74 [==============================] - 0s 142us/sample - loss: 0.0319 - accuracy: 0.8784
    Epoch 162/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0301 - accuracy: 0.8649
    Epoch 163/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0305 - accuracy: 0.8919
    Epoch 164/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0329 - accuracy: 0.8649
    Epoch 165/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0336 - accuracy: 0.8649
    Epoch 166/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0331 - accuracy: 0.8649
    Epoch 167/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0319 - accuracy: 0.8784
    Epoch 168/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0318 - accuracy: 0.8919
    Epoch 169/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0302 - accuracy: 0.8784
    Epoch 170/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0327 - accuracy: 0.8649
    Epoch 171/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0303 - accuracy: 0.8784
    Epoch 172/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0315 - accuracy: 0.8919
    Epoch 173/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0319 - accuracy: 0.8649
    Epoch 174/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0285 - accuracy: 0.8649
    Epoch 175/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0320 - accuracy: 0.8919
    Epoch 176/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 177/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0338 - accuracy: 0.8919
    Epoch 178/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0333 - accuracy: 0.8784
    Epoch 179/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0242 - accuracy: 0.9054
    Epoch 180/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0379 - accuracy: 0.8649
    Epoch 181/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0308 - accuracy: 0.8919
    Epoch 182/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0340 - accuracy: 0.8784
    Epoch 183/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0352 - accuracy: 0.8784
    Epoch 184/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0267 - accuracy: 0.9054
    Epoch 185/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0274 - accuracy: 0.8784
    Epoch 186/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0342 - accuracy: 0.8784
    Epoch 187/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0351 - accuracy: 0.8649
    Epoch 188/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0363 - accuracy: 0.8649
    Epoch 189/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0295 - accuracy: 0.8784
    Epoch 190/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 191/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0307 - accuracy: 0.8784
    Epoch 192/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0287 - accuracy: 0.8784
    Epoch 193/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0304 - accuracy: 0.8919
    Epoch 194/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0301 - accuracy: 0.8649
    Epoch 195/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0309 - accuracy: 0.8784
    Epoch 196/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0355 - accuracy: 0.8514
    Epoch 197/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0329 - accuracy: 0.8649
    Epoch 198/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0389 - accuracy: 0.8514
    Epoch 199/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0331 - accuracy: 0.8649
    Epoch 200/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0315 - accuracy: 0.8919
    Epoch 201/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0311 - accuracy: 0.8784
    Epoch 202/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0384 - accuracy: 0.8649
    Epoch 203/300
    74/74 [==============================] - 0s 104us/sample - loss: 0.0288 - accuracy: 0.8649
    Epoch 204/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0279 - accuracy: 0.8919
    Epoch 205/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0381 - accuracy: 0.8514
    Epoch 206/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0280 - accuracy: 0.8784
    Epoch 207/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0319 - accuracy: 0.8514
    Epoch 208/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0324 - accuracy: 0.8919
    Epoch 209/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0294 - accuracy: 0.8784
    Epoch 210/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0332 - accuracy: 0.8649
    Epoch 211/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0329 - accuracy: 0.8649
    Epoch 212/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0298 - accuracy: 0.8784
    Epoch 213/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0333 - accuracy: 0.8514
    Epoch 214/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0302 - accuracy: 0.8649
    Epoch 215/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0306 - accuracy: 0.8919
    Epoch 216/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0307 - accuracy: 0.8649
    Epoch 217/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0318 - accuracy: 0.8649
    Epoch 218/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0337 - accuracy: 0.8649
    Epoch 219/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0311 - accuracy: 0.8649
    Epoch 220/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0345 - accuracy: 0.8649
    Epoch 221/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0351 - accuracy: 0.8649
    Epoch 222/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0289 - accuracy: 0.8784
    Epoch 223/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0295 - accuracy: 0.8649
    Epoch 224/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0302 - accuracy: 0.8649
    Epoch 225/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0366 - accuracy: 0.8514
    Epoch 226/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0282 - accuracy: 0.8649
    Epoch 227/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 228/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0319 - accuracy: 0.8784
    Epoch 229/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0349 - accuracy: 0.8649
    Epoch 230/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0320 - accuracy: 0.8649
    Epoch 231/300
    74/74 [==============================] - 0s 125us/sample - loss: 0.0369 - accuracy: 0.8649
    Epoch 232/300
    74/74 [==============================] - 0s 104us/sample - loss: 0.0306 - accuracy: 0.8649
    Epoch 233/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0324 - accuracy: 0.8784
    Epoch 234/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0245 - accuracy: 0.8919
    Epoch 235/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0406 - accuracy: 0.8514
    Epoch 236/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0310 - accuracy: 0.8649
    Epoch 237/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0309 - accuracy: 0.8649
    Epoch 238/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0309 - accuracy: 0.8784
    Epoch 239/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0293 - accuracy: 0.8649
    Epoch 240/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0354 - accuracy: 0.8649
    Epoch 241/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0353 - accuracy: 0.8784
    Epoch 242/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0329 - accuracy: 0.8649
    Epoch 243/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0321 - accuracy: 0.8649
    Epoch 244/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0279 - accuracy: 0.8649
    Epoch 245/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0312 - accuracy: 0.8649
    Epoch 246/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0355 - accuracy: 0.8649
    Epoch 247/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0292 - accuracy: 0.8649
    Epoch 248/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0353 - accuracy: 0.8649
    Epoch 249/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0272 - accuracy: 0.9054
    Epoch 250/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0301 - accuracy: 0.8649
    Epoch 251/300
    74/74 [==============================] - 0s 119us/sample - loss: 0.0294 - accuracy: 0.8784
    Epoch 252/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0270 - accuracy: 0.8919
    Epoch 253/300
    74/74 [==============================] - 0s 117us/sample - loss: 0.0326 - accuracy: 0.8649
    Epoch 254/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0338 - accuracy: 0.8649
    Epoch 255/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0295 - accuracy: 0.8784
    Epoch 256/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0312 - accuracy: 0.8784
    Epoch 257/300
    74/74 [==============================] - 0s 124us/sample - loss: 0.0346 - accuracy: 0.8514
    Epoch 258/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 259/300
    74/74 [==============================] - 0s 107us/sample - loss: 0.0308 - accuracy: 0.8649
    Epoch 260/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0315 - accuracy: 0.8649
    Epoch 261/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0296 - accuracy: 0.8649
    Epoch 262/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0304 - accuracy: 0.8784
    Epoch 263/300
    74/74 [==============================] - 0s 110us/sample - loss: 0.0290 - accuracy: 0.8649
    Epoch 264/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0315 - accuracy: 0.8784
    Epoch 265/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0350 - accuracy: 0.8649
    Epoch 266/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0328 - accuracy: 0.8649
    Epoch 267/300
    74/74 [==============================] - 0s 127us/sample - loss: 0.0289 - accuracy: 0.8784
    Epoch 268/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0344 - accuracy: 0.8784
    Epoch 269/300
    74/74 [==============================] - 0s 126us/sample - loss: 0.0318 - accuracy: 0.8649
    Epoch 270/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0305 - accuracy: 0.8649
    Epoch 271/300
    74/74 [==============================] - 0s 173us/sample - loss: 0.0291 - accuracy: 0.8649
    Epoch 272/300
    74/74 [==============================] - 0s 153us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 273/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0316 - accuracy: 0.8649
    Epoch 274/300
    74/74 [==============================] - 0s 122us/sample - loss: 0.0281 - accuracy: 0.8649
    Epoch 275/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0291 - accuracy: 0.8649
    Epoch 276/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0311 - accuracy: 0.8649
    Epoch 277/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0309 - accuracy: 0.8919
    Epoch 278/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0338 - accuracy: 0.8649
    Epoch 279/300
    74/74 [==============================] - 0s 129us/sample - loss: 0.0329 - accuracy: 0.8784
    Epoch 280/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0331 - accuracy: 0.8919
    Epoch 281/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0332 - accuracy: 0.8649
    Epoch 282/300
    74/74 [==============================] - 0s 112us/sample - loss: 0.0272 - accuracy: 0.8784
    Epoch 283/300
    74/74 [==============================] - 0s 113us/sample - loss: 0.0310 - accuracy: 0.8784
    Epoch 284/300
    74/74 [==============================] - 0s 116us/sample - loss: 0.0309 - accuracy: 0.8784
    Epoch 285/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0332 - accuracy: 0.8649
    Epoch 286/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0340 - accuracy: 0.8649
    Epoch 287/300
    74/74 [==============================] - 0s 108us/sample - loss: 0.0311 - accuracy: 0.8649
    Epoch 288/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0328 - accuracy: 0.8649
    Epoch 289/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0349 - accuracy: 0.8649
    Epoch 290/300
    74/74 [==============================] - 0s 118us/sample - loss: 0.0357 - accuracy: 0.8649
    Epoch 291/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0334 - accuracy: 0.8784
    Epoch 292/300
    74/74 [==============================] - 0s 121us/sample - loss: 0.0343 - accuracy: 0.8649
    Epoch 293/300
    74/74 [==============================] - 0s 131us/sample - loss: 0.0305 - accuracy: 0.8649
    Epoch 294/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0313 - accuracy: 0.8649
    Epoch 295/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0342 - accuracy: 0.8649
    Epoch 296/300
    74/74 [==============================] - 0s 109us/sample - loss: 0.0319 - accuracy: 0.8649
    Epoch 297/300
    74/74 [==============================] - 0s 120us/sample - loss: 0.0323 - accuracy: 0.8649
    Epoch 298/300
    74/74 [==============================] - 0s 115us/sample - loss: 0.0349 - accuracy: 0.8649
    Epoch 299/300
    74/74 [==============================] - 0s 114us/sample - loss: 0.0316 - accuracy: 0.8649
    Epoch 300/300
    74/74 [==============================] - 0s 111us/sample - loss: 0.0330 - accuracy: 0.8649



```python
print ('Average f1 score', np.mean(test_F1))
print ('Average Run time', np.mean(time_k))
```

    Average f1 score 0.6
    Average Run time 3.3290751775105796


#### Building an LSTM Classifier on the sequences for comparison
We built an LSTM Classifier on the sequences to compare the accuracy.


```python
X = darpa_data['seq']
encoded_X = np.ndarray(shape=(len(X),), dtype=list)
for i in range(0,len(X)):
    encoded_X[i]=X.iloc[i].split("~")
```


```python
max_seq_length = np.max(darpa_data['seqlen'])
encoded_X = tf.keras.preprocessing.sequence.pad_sequences(encoded_X, maxlen=max_seq_length)
```


```python
kfold = 3
random_state = 11

test_F1 = np.zeros(kfold)
time_k = np.zeros(kfold)

epochs = 50
batch_size = 15
skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)
k = 0

for train_index, test_index in skf.split(encoded_X, y):
    X_train, X_test = encoded_X[train_index], encoded_X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    embedding_vecor_length = 32
    top_words=50
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_seq_length))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    start_time = time.time()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    end_time=time.time()
    time_k[k]=end_time-start_time

    y_pred = model.predict_proba(X_test).round().astype(int)
    y_train_pred=model.predict_proba(X_train).round().astype(int)
    test_F1[k]=sklearn.metrics.f1_score(y_test, y_pred)
    k+=1
```

    Model: "sequential_13"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 1773, 32)          1600      
    _________________________________________________________________
    lstm (LSTM)                  (None, 32)                8320      
    _________________________________________________________________
    dense_36 (Dense)             (None, 1)                 33        
    _________________________________________________________________
    activation_36 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 9,953
    Trainable params: 9,953
    Non-trainable params: 0
    _________________________________________________________________
    Train on 74 samples
    Epoch 1/50
    74/74 [==============================] - 4s 60ms/sample - loss: 0.6934 - accuracy: 0.5135
    Epoch 2/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.6591 - accuracy: 0.8784
    Epoch 3/50
    74/74 [==============================] - 3s 46ms/sample - loss: 0.6201 - accuracy: 0.8784
    Epoch 4/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.5612 - accuracy: 0.8784
    Epoch 5/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.4500 - accuracy: 0.8784
    Epoch 6/50
    74/74 [==============================] - 3s 46ms/sample - loss: 0.3808 - accuracy: 0.8784
    Epoch 7/50
    74/74 [==============================] - 4s 49ms/sample - loss: 0.3807 - accuracy: 0.8784
    Epoch 8/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3795 - accuracy: 0.8784
    Epoch 9/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3718 - accuracy: 0.8784
    Epoch 10/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3713 - accuracy: 0.8784
    Epoch 11/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.3697 - accuracy: 0.8784
    Epoch 12/50
    74/74 [==============================] - 3s 46ms/sample - loss: 0.3696 - accuracy: 0.8784
    Epoch 13/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3696 - accuracy: 0.8784
    Epoch 14/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3677 - accuracy: 0.8784
    Epoch 15/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3666 - accuracy: 0.8784
    Epoch 16/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3661 - accuracy: 0.8784
    Epoch 17/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3654 - accuracy: 0.8784
    Epoch 18/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3634 - accuracy: 0.8784
    Epoch 19/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3638 - accuracy: 0.8784
    Epoch 20/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3598 - accuracy: 0.8784
    Epoch 21/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3584 - accuracy: 0.8784
    Epoch 22/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3539 - accuracy: 0.8784
    Epoch 23/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3588 - accuracy: 0.8784
    Epoch 24/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3374 - accuracy: 0.8784
    Epoch 25/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3356 - accuracy: 0.8784
    Epoch 26/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3044 - accuracy: 0.8784
    Epoch 27/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.2896 - accuracy: 0.8784
    Epoch 28/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2864 - accuracy: 0.8784
    Epoch 29/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2430 - accuracy: 0.8784
    Epoch 30/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2675 - accuracy: 0.8784
    Epoch 31/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2764 - accuracy: 0.8784
    Epoch 32/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2404 - accuracy: 0.8784
    Epoch 33/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2131 - accuracy: 0.8784
    Epoch 34/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2109 - accuracy: 0.8784
    Epoch 35/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2060 - accuracy: 0.8919
    Epoch 36/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1925 - accuracy: 0.9054
    Epoch 37/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1913 - accuracy: 0.9189
    Epoch 38/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1947 - accuracy: 0.9324
    Epoch 39/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1762 - accuracy: 0.9324
    Epoch 40/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1856 - accuracy: 0.9459
    Epoch 41/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1689 - accuracy: 0.9324
    Epoch 42/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1762 - accuracy: 0.9324
    Epoch 43/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.1914 - accuracy: 0.9459
    Epoch 44/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.1867 - accuracy: 0.9595
    Epoch 45/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.1602 - accuracy: 0.9459
    Epoch 46/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1627 - accuracy: 0.9324
    Epoch 47/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1475 - accuracy: 0.9595
    Epoch 48/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1527 - accuracy: 0.9595
    Epoch 49/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1408 - accuracy: 0.9595
    Epoch 50/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1745 - accuracy: 0.9595
    Model: "sequential_14"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 1773, 32)          1600      
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 32)                8320      
    _________________________________________________________________
    dense_37 (Dense)             (None, 1)                 33        
    _________________________________________________________________
    activation_37 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 9,953
    Trainable params: 9,953
    Non-trainable params: 0
    _________________________________________________________________
    Train on 74 samples
    Epoch 1/50
    74/74 [==============================] - 4s 59ms/sample - loss: 0.6898 - accuracy: 0.5676
    Epoch 2/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.6513 - accuracy: 0.8784
    Epoch 3/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.6120 - accuracy: 0.8784
    Epoch 4/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.5458 - accuracy: 0.8784
    Epoch 5/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.4240 - accuracy: 0.8784
    Epoch 6/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3963 - accuracy: 0.8784
    Epoch 7/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3924 - accuracy: 0.8784
    Epoch 8/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.3851 - accuracy: 0.8784
    Epoch 9/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3731 - accuracy: 0.8784
    Epoch 10/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3708 - accuracy: 0.8784
    Epoch 11/50
    74/74 [==============================] - 3s 46ms/sample - loss: 0.3737 - accuracy: 0.8784
    Epoch 12/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.3716 - accuracy: 0.8784
    Epoch 13/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3706 - accuracy: 0.8784
    Epoch 14/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3697 - accuracy: 0.8784
    Epoch 15/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3698 - accuracy: 0.8784
    Epoch 16/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.3686 - accuracy: 0.8784
    Epoch 17/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3686 - accuracy: 0.8784
    Epoch 18/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3682 - accuracy: 0.8784
    Epoch 19/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.3667 - accuracy: 0.8784
    Epoch 20/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3678 - accuracy: 0.8784
    Epoch 21/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3640 - accuracy: 0.8784
    Epoch 22/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3621 - accuracy: 0.8784
    Epoch 23/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3601 - accuracy: 0.8784
    Epoch 24/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3574 - accuracy: 0.8784
    Epoch 25/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3514 - accuracy: 0.8784
    Epoch 26/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3552 - accuracy: 0.8784
    Epoch 27/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3381 - accuracy: 0.8784
    Epoch 28/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3274 - accuracy: 0.8784
    Epoch 29/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3118 - accuracy: 0.8784
    Epoch 30/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2943 - accuracy: 0.8784
    Epoch 31/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.2783 - accuracy: 0.8784
    Epoch 32/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.2459 - accuracy: 0.8784
    Epoch 33/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.2276 - accuracy: 0.8919
    Epoch 34/50
    74/74 [==============================] - 3s 46ms/sample - loss: 0.2345 - accuracy: 0.9189
    Epoch 35/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.1888 - accuracy: 0.9189
    Epoch 36/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.2413 - accuracy: 0.9189
    Epoch 37/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.2389 - accuracy: 0.8649
    Epoch 38/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.2136 - accuracy: 0.9054
    Epoch 39/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.1933 - accuracy: 0.9054
    Epoch 40/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.1882 - accuracy: 0.8919
    Epoch 41/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.1999 - accuracy: 0.9054
    Epoch 42/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1760 - accuracy: 0.8919
    Epoch 43/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.1990 - accuracy: 0.8243
    Epoch 44/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1632 - accuracy: 0.9189
    Epoch 45/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1626 - accuracy: 0.9189
    Epoch 46/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1700 - accuracy: 0.8784
    Epoch 47/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1529 - accuracy: 0.9189
    Epoch 48/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1641 - accuracy: 0.9189
    Epoch 49/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1482 - accuracy: 0.9189
    Epoch 50/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.1661 - accuracy: 0.8784
    Model: "sequential_15"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 1773, 32)          1600      
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 32)                8320      
    _________________________________________________________________
    dense_38 (Dense)             (None, 1)                 33        
    _________________________________________________________________
    activation_38 (Activation)   (None, 1)                 0         
    =================================================================
    Total params: 9,953
    Trainable params: 9,953
    Non-trainable params: 0
    _________________________________________________________________
    Train on 74 samples
    Epoch 1/50
    74/74 [==============================] - 5s 63ms/sample - loss: 0.6756 - accuracy: 0.8919
    Epoch 2/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.6397 - accuracy: 0.8919
    Epoch 3/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.5892 - accuracy: 0.8919
    Epoch 4/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.5005 - accuracy: 0.8919
    Epoch 5/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3800 - accuracy: 0.8919
    Epoch 6/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.3459 - accuracy: 0.8919
    Epoch 7/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.3529 - accuracy: 0.8919
    Epoch 8/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3502 - accuracy: 0.8919
    Epoch 9/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.3455 - accuracy: 0.8919
    Epoch 10/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3438 - accuracy: 0.8919
    Epoch 11/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3434 - accuracy: 0.8919
    Epoch 12/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3431 - accuracy: 0.8919
    Epoch 13/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.3433 - accuracy: 0.8919
    Epoch 14/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3433 - accuracy: 0.8919
    Epoch 15/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.3432 - accuracy: 0.8919
    Epoch 16/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.3421 - accuracy: 0.8919
    Epoch 17/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3426 - accuracy: 0.8919
    Epoch 18/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3423 - accuracy: 0.8919
    Epoch 19/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3424 - accuracy: 0.8919
    Epoch 20/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.3420 - accuracy: 0.8919
    Epoch 21/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3429 - accuracy: 0.8919
    Epoch 22/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.3412 - accuracy: 0.8919
    Epoch 23/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3402 - accuracy: 0.8919
    Epoch 24/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.3397 - accuracy: 0.8919
    Epoch 25/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.3390 - accuracy: 0.8919
    Epoch 26/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3398 - accuracy: 0.8919
    Epoch 27/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.3372 - accuracy: 0.8919
    Epoch 28/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3374 - accuracy: 0.8919
    Epoch 29/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3323 - accuracy: 0.8919
    Epoch 30/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.3323 - accuracy: 0.8919
    Epoch 31/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.3253 - accuracy: 0.8919
    Epoch 32/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.3228 - accuracy: 0.8919
    Epoch 33/50
    74/74 [==============================] - 3s 40ms/sample - loss: 0.3075 - accuracy: 0.8919
    Epoch 34/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2985 - accuracy: 0.8919
    Epoch 35/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.3000 - accuracy: 0.8919
    Epoch 36/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.2791 - accuracy: 0.8919
    Epoch 37/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.2580 - accuracy: 0.8919
    Epoch 38/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.2874 - accuracy: 0.8919
    Epoch 39/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.2712 - accuracy: 0.8919
    Epoch 40/50
    74/74 [==============================] - 3s 44ms/sample - loss: 0.2432 - accuracy: 0.8919
    Epoch 41/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.2231 - accuracy: 0.8919
    Epoch 42/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.2146 - accuracy: 0.8919
    Epoch 43/50
    74/74 [==============================] - 3s 45ms/sample - loss: 0.2026 - accuracy: 0.8919
    Epoch 44/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.2371 - accuracy: 0.9054
    Epoch 45/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.2293 - accuracy: 0.9189
    Epoch 46/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2524 - accuracy: 0.9324
    Epoch 47/50
    74/74 [==============================] - 3s 42ms/sample - loss: 0.2331 - accuracy: 0.9189
    Epoch 48/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.2046 - accuracy: 0.8919
    Epoch 49/50
    74/74 [==============================] - 3s 43ms/sample - loss: 0.2020 - accuracy: 0.8919
    Epoch 50/50
    74/74 [==============================] - 3s 41ms/sample - loss: 0.1992 - accuracy: 0.9054



```python
print ('Average f1 score', np.mean(test_F1))
print ('Average Run time', np.mean(time_k))
```

    Average f1 score 0.3313492063492064
    Average Run time 157.32080109914145


We find that the LSTM classifier gives a significantly lower F1 score. This may be improved by changing the model. However, we find that the SGT embedding could work with a small and unbalanced data without the need of a complicated classifier model.

LSTM models typically require more data for training and also has significantly more computation time. The LSTM model above took 425.6 secs while the MLP model took just 9.1 secs.


```python

```
