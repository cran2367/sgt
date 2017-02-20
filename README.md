# Sequence Graph Transform (SGT)


This is open source code repository for SGT. Sequence Graph Transform extracts the short- and long-term sequence features and embeds them in a finite-dimensional feature space. Importantly, SGT has low computation and can extract any amount of short- to long-term patterns without any increase in the computation. These properties are proved theoretically and demonstrated on real data in this paper: https://arxiv.org/abs/1608.03533.

If using this code, please cite the following:

Ranjan, Chitta, Samaneh Ebrahimi, and Kamran Paynabar. "Sequence Graph Transform (SGT): A Feature Extraction Function for Sequence Data Mining." arXiv preprint arXiv:1608.03533 (2016).

@article{ranjan2016sequence,
  title={Sequence Graph Transform (SGT): A Feature Extraction Function for Sequence Data Mining},
  author={Ranjan, Chitta and Ebrahimi, Samaneh and Paynabar, Kamran},
  journal={arXiv preprint arXiv:1608.03533},
  year={2016}
}

## Quick validation of your code
For a sample sequence, `BBACACAABA`, the parts of SGT, W<sup>(0)</sup> and W<sup>(\kappa)</sup>
