import numpy as np
import pandas as pd
from itertools import chain
from itertools import product as iterproduct
import warnings

class SGT():
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

    Parameters
    ----------
    Input

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

    lengthsensitive Default False. This is set to true if the embedding of
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

    mode            Choices in {'default', 'multiprocessing'}.

    processors      Used if mode is 'multiprocessing'. By default, the
                    number of processors used in multiprocessing is
                    number of available - 1.
    '''

    def __init__(self,
                 alphabets=[],
                 kappa=1,
                 lengthsensitive=False,
                 flatten=True,
                 mode='default',
                 processors=None,
                 lazy=False):

        self.alphabets = alphabets
        
        if len(self.alphabets) != 0:
            self.feature_names = self.__set_feature_names(self.alphabets)
        
        self.kappa = kappa
        self.lengthsensitive = lengthsensitive
        self.flatten = flatten
        self.mode = mode
        self.processors = processors

        if self.processors==None:
            import os
            self.processors = os.cpu_count() - 1

        self.lazy = lazy

    def getpositions(self, sequence, alphabets):
        '''
        Compute index position elements in the sequence
        given alphabets set.

        Return list of tuples [(value, position)]
        '''
        positions = [(v, np.where(sequence == v))
                     for v in alphabets if v in sequence]

        return positions

    def fit(self, sequence):
        '''
        Extract Sequence Graph Transform features using Algorithm-2.

        sequence            An array of discrete elements. For example,
                            np.array(["B","B","A","C","A","C","A","A","B","A"].

        return: sgt matrix or vector (depending on Flatten==False or True)

        '''

        sequence = np.array(sequence)

        if(len(self.alphabets) == 0):
            self.alphabets = self.estimate_alphabets(sequence)
            self.feature_names = self.__set_feature_names(self.alphabets)
            
        size = len(self.alphabets)
        l = 0
        W0, Wk = np.zeros((size, size)), np.zeros((size, size))
        positions = self.getpositions(sequence, self.alphabets)

        alphabets_in_sequence = np.unique(sequence)

        for i, u in enumerate(alphabets_in_sequence):
            index = [p[0] for p in positions].index(u)

            U = np.array(positions[index][1]).ravel()

            for j, v in enumerate(alphabets_in_sequence):
                index = [p[0] for p in positions].index(v)

                V2 = np.array(positions[index][1]).ravel()

                C = [(i, j) for i in U for j in V2 if j > i]

                cu = np.array([ic[0] for ic in C])
                cv = np.array([ic[1] for ic in C])

                # Insertion positions
                pos_i = self.alphabets.index(u)
                pos_j = self.alphabets.index(v)

                W0[pos_i, pos_j] = len(C)

                Wk[pos_i, pos_j] = np.sum(np.exp(-self.kappa * np.abs(cu - cv)))

            l += U.shape[0]

        if self.lengthsensitive:
            W0 /= l

        W0[np.where(W0 == 0)] = 1e7  # avoid divide by 0

        sgt = np.power(np.divide(Wk, W0), 1/self.kappa)

        if(self.flatten):
            sgt = pd.Series(sgt.flatten(), index=self.feature_names)
        else:
            sgt = pd.DataFrame(sgt, 
                               columns=self.alphabets, 
                               index=self.alphabets)
        return sgt

    def __flatten(self, listOfLists):
        "Flatten one level of nesting"
        flat = [x for sublist in listOfLists for x in sublist]
        return flat

    def estimate_alphabets(self, corpus):
        if len(corpus) > 1e5:
            print("Error: Too many sequences. Pass the alphabet list as an input. Exiting.")
            sys.exit(1)
        else:
            return(np.unique(np.asarray(self.__flatten(corpus))).tolist())

    def set_alphabets(self, corpus):
        self.alphabets = self.estimate_alphabets(corpus)
        self.feature_names = self.__set_feature_names(self.alphabets)
        return self

    def get_alphabets(self):
        return self.alphabets
    
    def get_feature_names(self):
        return self.feature_names

    def __fit_to_list(self, sequence):
        return list(self.fit(sequence))

    def __set_feature_names(self, alphabets):
        return list(iterproduct(alphabets, alphabets))

    def fit_transform(self, corpus):
        '''
        Inputs:
        corpus       A list of sequences. Each sequence is a list of alphabets.
        '''

        if(len(self.alphabets) == 0):
            self.alphabets = self.estimate_alphabets(corpus['sequence'])
            self.feature_names = self.__set_feature_names(self.alphabets)

        if self.mode=='default':
            sgt = corpus.apply(lambda x: [x['id']] + list(self.fit(x['sequence'])), 
                               axis=1, 
                               result_type='expand')
            sgt.columns = ['id'] + self.feature_names
            return sgt
        elif self.mode=='multiprocessing':
            # Import
            from pandarallel import pandarallel
            # Initialization
            pandarallel.initialize(nb_workers=self.processors)
            sgt = corpus.parallel_apply(lambda x: [x['id']] + 
                                        list(self.fit(x['sequence'])), 
                                        axis=1, 
                                        result_type='expand')
            sgt.columns = ['id'] + self.feature_names            
            return sgt
    
    def transform(self, corpus):
        '''
        Inputs:
        corpus       A list of sequences. Each sequence is a list of alphabets.
        '''
        
        '''
        Difference between fit_transform and transform is:
        In transform() we have the alphabets already  known.
        In fit_transform() is alphabets are not known, they
        are computed.
        The computation in fit is essentially getting the
        alphabets set.
        '''

        if self.mode=='default':
            sgt = corpus.apply(lambda x: [x['id']] + list(self.fit(x['sequence'])), 
                               axis=1, 
                               result_type='expand')
            sgt.columns = ['id'] + self.feature_names
            return sgt
        elif self.mode=='multiprocessing':
            # Import
            from pandarallel import pandarallel
            # Initialization
            pandarallel.initialize(nb_workers=self.processors)
            sgt = corpus.parallel_apply(lambda x: [x['id']] + 
                                        list(self.fit(x['sequence'])), 
                                        axis=1, 
                                        result_type='expand')
            sgt.columns = ['id'] + self.feature_names            
            return sgt