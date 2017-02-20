
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Sequence Graph Transform (SGT)

#### Author: Chitta Ranjan

This is open source code repository for SGT. Sequence Graph Transform extracts the short- and long-term sequence features and embeds them in a finite-dimensional feature space. Importantly, SGT has low computation and can extract any amount of short- to long-term patterns without any increase in the computation. These properties are proved theoretically and demonstrated on real data in this paper: https://arxiv.org/abs/1608.03533.

If using this code, please cite the following:

[1] Ranjan, Chitta, Samaneh Ebrahimi, and Kamran Paynabar. "Sequence Graph Transform (SGT): A Feature Extraction Function for Sequence Data Mining." arXiv preprint arXiv:1608.03533 (2016).

@article{ranjan2016sequence,
  title={Sequence Graph Transform (SGT): A Feature Extraction Function for Sequence Data Mining},
  author={Ranjan, Chitta and Ebrahimi, Samaneh and Paynabar, Kamran},
  journal={arXiv preprint arXiv:1608.03533},
  year={2016}
}

## Quick validation of your code
Apply the algorithm on a sequence `BBACACAABA`. The parts of SGT, W<sup>(0)</sup> and W<sup>(\kappa)</sup>, in Algorithm 1 & 2 in [1], and the resulting SGT estimate will be:

```
seq <- "BBACACAABA"
kappa <- 5
###### Algorithm 1 ######
sgt_parts_alg1 <- f_sgt_parts(sequence = seq, kappa = kappa, alphabet_set_size = length(c("A","B","C"))
```

```
$W0
   A B C
A 10 4 3
B 11 3 4
C  7 2 1

$W_kappa
            A            B            C
A 0.006874761 6.783349e-03 1.347620e-02
B 0.013521602 6.737947e-03 4.570791e-05
C 0.013521604 3.059162e-07 4.539993e-05
```

```
sgt <- f_SGT(W_kappa = sgt_parts_alg1$W_kappa, W0 = sgt_parts_alg1$W0, 
             Len = sgt_parts_alg1$Len, kappa = kappa)  # Set Len = NULL for length-sensitive SGT.
print(sgt)
```

```
          A          B         C
A 0.3693614 0.44246287 0.5376371
B 0.4148844 0.46803816 0.1627745
C 0.4541361 0.06869332 0.2144920

```

## Using the code
Open file `main.R` and execute line-by-line to understand the process. In this sample execution, we present SGT estimation from either of the two algorithms presented in [1]. The first part is for understanding the SGT computation process.

In the next part we demonstrate sequence clustering using SGT on a synthesized sample dataset. The sequence lengths in the dataset ranges between (45, 711) with a uniform distribution (hence, average length is ~365). Similar sequences in the dataset has some common substrings. This common substrings can be of any length. Also, the order of the instances of these substrings is arbitrary and random in different sequences. For example,
```AKQZTAEEYTDZUXXIRZSTAYFUIXCPDZUXMCSMEMVDVGMTDRDDEJWNDGDPSVPKJHKQBRKMXHHNLUBXBMHISQWEHGXGDDCADPVKESYQXGRLRZSTAYFUOQZTAWTBRKMXHHNWYRYBRKMXHHNPRNRBRKMXHHNPBMHIUSVXBMHIWXQRZSTAYFUCWRZSTAYFUJEJDZUXPUEMVDVGMTOHUDZUXLOQSKESYQXGRCTLBRKMXHHNNJZDZUXTFWZKESYQXGRUATSNDGDPWEBNIQZMBNIQKESYQXGRSZTTPTZWRMEMVDVGMTAPBNIRPSADZUXJTEDESOKPTLJEMZTDLUIPSMZTDLUIWYDELISBRKMXHHNMADEDXKESYQXGRWEFRZSTAYFUDNDGDPKYEKPTSXMKNDGDPUTIQJHKSDZUXVMZTDLUINFNDGDPMQZTAPPKBMHIUQIUBMHIEKKJHK
```

```
SDBRKMXHHNRATBMHIYDZUXMTRMZTDLUIEKDEIBQZTAZOAMZTDLUILHGXGDDCAZEXJHKTDOOHGXGDDCAKZHNEMVDVGMTIHZXDEROEQDEGZPPTDBCLBMHIJMMKESYQXGRGDPTNBRKMXHHNGCBYNDGDPKMWKBMHIDQDZUXIHKVBMHINQZTAHBRKMXHHNIRBRKMXHHNDISDZUXWBOYEMVDVGMTNTAQZTA
```
The average noise in the sequences is about 40%. A noise means
