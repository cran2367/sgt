import numpy as np
import pandas as pd
from itertools import chain
import warnings


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

    Parameters
    ----------
    Input

    alphabets       Optional. The set of alphabets that make up all 
                    the sequences in the dataset. If not passed, the
                    alphabet set is automatically computed as the 
                    unique set of elements that make all the sequences.

    kappa           Tuning parameter, kappa > 0, to change the extraction of 
                    long-term dependency. Higher the value the lesser
                    the long-term dependency captured in the embedding.
                    Typical values for kappa are 1, 5, 10.

    lengthsensitive Default false. This is set to true if the embedding of
                    should have the information of the length of the sequence.
                    If set to false then the embedding of two sequences with
                    similar pattern but different lengths will be the same.
                    lengthsensitive = false is similar to length-normalization.
    '''

    def __init__(self, alphabets=[], kappa=1, lengthsensitive=True):
        self.alphabets = alphabets
        self.kappa = kappa
        self.lengthsensitive = lengthsensitive

    def getpositions(self, sequence, alphabets):
        '''
        Compute index position elements in the sequence
        given alphabets set.

        Return list of tuples [(value, position)]
        '''
        positions = [(v, np.where(sequence == v))
                     for v in alphabets if v in sequence]

        return positions

    def fit(self, sequence, alphabets, lengthsensitive, kappa, flatten):
        '''
        Extract Sequence Graph Transform features using Algorithm-2.

        sequence            An array of discrete elements. For example,
                            np.array(["B","B","A","C","A","C","A","A","B","A"].
        alphabets           An array of the set of elements that make up the sequences.
                            For example, np.array(["A", "B", "C"].
        lengthsensitive     A boolean set to false by default.
        kappa               A positive value tuning parameter that is by default 1. 
                            Typically selected kappa from {1, 5, 10}.
        flatten

        return: sgt matrix

        '''
        
        size = len(alphabets)
        l = 0
        W0, Wk = np.zeros((size, size)), np.zeros((size, size))
        positions = self.getpositions(sequence, alphabets)

        alphabets_in_sequence = np.unique(sequence)

        for i, u in enumerate(alphabets_in_sequence):
            index = [p[0] for p in positions].index(u)

            U = np.array(positions[index][1]).ravel()

            for j, v in enumerate(alphabets_in_sequence):
                index = [p[0] for p in positions].index(v)

                V2 = np.array(positions[index][1]).ravel()

                C = [(i, j) for i in U for j in V2 if j > i]

                cu = np.array([i[0] for i in C])
                cv = np.array([i[1] for i in C])

                # Insertion positions
                pos_i = alphabets.index(u)
                pos_j = alphabets.index(v)

                W0[pos_i, pos_j] = len(C)

                Wk[pos_i, pos_j] = np.sum(np.exp(-kappa * np.abs(cu - cv)))

            l += U.shape[0]

        if lengthsensitive:
            W0 /= l

        W0[np.where(W0 == 0)] = 1e7  # avoid divide by 0

        sgt = np.power(np.divide(Wk, W0), 1/kappa)

        if(flatten):
            sgt = sgt.flatten()
        else:
            sgt = pd.DataFrame(sgt)
            sgt.columns = alphabets
            sgt.index = alphabets

        return sgt

    def __flatten(self, listOfLists):
        "Flatten one level of nesting"
        flat = [x for sublist in listOfLists for x in sublist]
        return flat

    def __estimate_alphabets(self, corpus):
        return(np.unique(np.asarray(self.__flatten(corpus))).tolist())

    def set_alphabets(self, corpus):
        self.alphabets = self.__estimate_alphabets(corpus)
        return self

    def fit_transform(self, corpus, flatten=True):
        '''
        corpus       A list of sequences. Each sequence is a list of alphabets.
        '''
        if(len(self.alphabets) == 0):
            self.alphabets = self.__estimate_alphabets(corpus)

        sgt = [self.fit(sequence=np.array(sequence), alphabets=self.alphabets,
                        lengthsensitive=self.lengthsensitive, kappa=self.kappa,
                        flatten=flatten).tolist() for sequence in corpus]

        return(np.array(sgt))

    def transform(self, corpus, flatten=True):
        '''
        Fit on sequence corpus that already fitted. The alphabets set should
        already be initialized.
        '''
        sgt = [self.fit(sequence=np.array(sequence), alphabets=self.alphabets,
                        lengthsensitive=self.lengthsensitive, kappa=self.kappa,
                        flatten=flatten).tolist() for sequence in corpus]

        return(np.array(sgt))