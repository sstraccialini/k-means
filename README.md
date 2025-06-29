# An exploration of novel heuristics for efficient and accurate $k$-means clustering

This repository contains the code written to develop and test the algorithms described in my Bachelor's final thesis.

## Table of Contents
- [An exploration of novel heuristics for efficient and accurate $k$-means clustering](#an-exploration-of-novel-heuristics-for-efficient-and-accurate-k-means-clustering)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Repository Structure](#repository-structure)
  - [Extended-Hartigan pseudocode](#extended-hartigan-pseudocode)
  - [Datasets](#datasets)
  - [References](#references)

## Abstract

$K$-means is one of the most widely-used algorithms to perform clustering and exploratory data analysis, since it allows identifying relevant patterns in data. Its objective is to partition a set of observations into a predetermined number of clusters, by minimizing the within-cluster sum of squares (WCSS). However, this optimization problem is NP-hard, necessitating the use of heuristic algorithms for practical applications. The most common heuristic, Lloyd's algorithm, is simple and efficient, but at the same time it is highly susceptible to converging to suboptimal local minima, with its performance being heavily dependent on the centroids' initial positions. While alternatives, such as Hartigan's algorithm, have been proven to find better (lower-cost) solutions, they are often significantly slower and more computationally intensive.

This thesis introduces and evaluates a novel heuristic, "extended-Hartigan", designed to bridge this gap by reducing the computational complexity of the standard Hartigan's method while retaining its ability to find high-quality clusterings. The proposed algorithm deviates from the standard single-point update rule. Instead, it first identifies all candidate data points whose reassignment would individually decrease the total cost, following Hartigan's procedure, and then attempts to apply this entire list of reassignments in a single "unsafe" batch update. If this aggressive update fails to lower the overall cost, the algorithm reverts the changes and proceeds in a "safe" mode, accepting a limited subset of non-conflicting reassignments to guarantee a monotonic cost reduction in that iteration.

To validate this approach, we conduct an empirical study comparing extended-Hartigan and its hybrid "mixed-mode" variants against the classical Lloyd's and Hartigan's algorithms. The evaluation spans several benchmark datasets, which feature diverse numbers of samples, dimensions, and cluster counts. The comparison is performed using multiple initialization techniques, including maximin, $k$-means++, and greedy $k$-means++, and performance is assessed based on the final cost, computational efficiency, and clustering stability. The results demonstrate that the proposed methods consistently find clusterings with a lower final cost than Lloyd's algorithm, achieving a significant cost reduction on the most complex datasets. Moreover, extended-Hartigan avoids the computational instability inherent in the standard Hartigan algorithm, showing a cost profile closer to the efficient Lloyd's baseline and successfully preventing extreme cost increases. The findings indicate that the extended-Hartigan algorithm is a robust and efficient alternative that successfully balances the trade-off between solution quality and computational expense, offering a practical and powerful alternative for $k$-means clustering.

## Repository Structure
Here's the structure of this repository:

- `data`: contains some datasets on which the algorithms where tested.
- `iter_records`: contains some records of safe and unsafe iterations using our proposed `extended-hartigan` algorithm, mainly for an illustrative scope.
- `latex`: contains the latex-formatted tables of results which are included in the Appendix of the final work.
- `misc`: contains a bunch of experimental code, proofs-of-concept (PoCs), and isolated tests developed during the main project's lifecycle. The code within this folder is not part of the final application, is not actively maintained, and can be safely ignored for any production build. It serves as a development 'sandbox' for trying out new ideas.
- `profiling`: contains some profiling code to speed up the algorithms; used during development phase.
- `tests`: contains all tests results.
- `utils`: contains some other useful code which is not directly related with the developed algorithms (such as code for results visualization and so on...).
- `Abstract.txt`: the abstract of the final work
- **`kmeans.ipynb`: the actual notebook containing all algorithms developed.**
- `Tesi-final.pdf`: the final version of my thesis work.

## Extended-Hartigan pseudocode

One of the main contribution of this work is the algorithm we called `extended-Hartigan`. Here follows its pseudocode.

```
choose k = number of centroids

assign points to each centroid randomly
move the centroid to the mean of points assigned to it

repeat:
    for each datapoint d ∈ c
        for each centroid c' ≠ c
            compute Δcost(d; c → c')
            | if result is negative, store in a candidates list
            | if candidate is already in list keep the reassignment with minimum Δcost
        end for
    end for

    if "unsafe mode":
        accept all candidates
        if overall cost did not decrease
            revert changes and proceed in "safe mode"
    else if "safe mode"
        sort all candidates (minimum first)
        accept them in order if the corresponding clusters have not been involved
 
until convergence
```

## Datasets

The main datasets used in this work where *A-Sets* from a research project by Karkkainen
and Franti [KF02], Bridge and House, from a work by Franti, Rezaei, and Zhao [FRZ14]. These, and other datasets which are used for clustering tasks, are available in the following website: https://cs.joensuu.fi/sipu/datasets/.

## References

[AV07] David Arthur and Sergei Vassilvitskii. “K-Means++: The Advantages of Careful
Seeding”. In: vol. 8. Jan. 2007, pp. 1027–1035. doi: 10.1145/1283383.1283494.

[Bha+19] Anup Bhattacharya et al. Noisy, Greedy and Not So Greedy k-means++. Dec.
2019. doi: 10.48550/arXiv.1912.00653. url: http://arxiv.org/abs/1912.00653.

[Elk03] Charles Elkan. “Using the triangle inequality to accelerate k-means”. In: Proceed-
ings of the Twentieth International Conference on International Conference on
Machine Learning. ICML’03. Washington, DC, USA: AAAI Press, 2003, pp. 147–
153. isbn: 1577351894.

[FRZ14] Pasi Franti, Mohammad Rezaei, and Qinpei Zhao. “Centroid index: Cluster level
similarity measure”. en. In: Pattern Recognition 47.9 (Sept. 2014), pp. 3034–
3045. issn: 00313203. doi: 10 . 1016 / j . patcog . 2014 . 03 . 017. url: https://linkinghub.elsevier.com/retrieve/pii/S0031320314001150.

[Gru+22] Christoph Grunau et al. A Nearly Tight 0Analysis of Greedy k-means++. July
2022. doi: 10.48550/arXiv.2207.07949. url: http://arxiv.org/abs/2207.07949.

[Har75] John A. Hartigan. Clustering algorithms. eng. A Wiley publication in applied
statistics. New York: Wiley, 1975. isbn: 9780471356455.

[HW79] J. A. Hartigan and M. A. Wong. “Algorithm AS 136: A K-Means Clustering Algo-
rithm”. In: Journal of the Royal Statistical Society. Series C (Applied Statistics)
28.1 (1979), pp. 100–108. issn: 00359254, 14679876. url: http://www.jstor.org/stable/2346830.

[KF02] Ismo Karkkainen and Pasi Franti. Dynamic local search algorithm for the cluster-
ing problem. eng. Report series / University of Joensuu, Department of Computer
Science. A, 2002-6. OCLC: 58380784. Joensuu: University of Joensuu, 2002. isbn:
9789524581431.

[Llo82] S. Lloyd. “Least squares quantization in PCM”. en. In: IEEE Transactions on
Information Theory 28.2 (Mar. 1982), pp. 129–137. issn: 0018-9448. doi: 10.
1109 / TIT . 1982 . 1056489. url: http://ieeexplore.ieee.org/document/1056489/.

[NF16] James Newling and Francois Fleuret. “Fast k-means with accurate bounds”.
en. In: Proceedings of The 33rd International Conference on Machine Learning.
PMLR, June 2016, pp. 936–944. url: https://proceedings.mlr.press/v48/newling16.html.

[SB14] Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: from
theory to algorithms. eng. New York: Cambridge university press, 2014. isbn:
9781107057135.

[Sci] Scikit-learn. KMeans. en. url: https://scikit-learn/stable/modules/generated/sklearn.cluster.KMeans.html.

[TV10] Matus Telgarsky and Andrea Vattani. “Hartigan’s Method: k-means Clustering
without Voronoi”. In: Proceedings of the Thirteenth International Conference on
Artificial Intelligence and Statistics. Ed. by Yee Whye Teh and Mike Titterington.
Vol. 9. Proceedings of Machine Learning Research. Chia Laguna Resort, Sardinia,
Italy: PMLR, May 2010, pp. 820–827. url: https://proceedings.mlr.press/v9/telgarsky10a.html.

[Wu+08] Xindong Wu et al. “Top 10 algorithms in data mining”. en. In: Knowledge and
Information Systems 14.1 (Jan. 2008), pp. 1–37. issn: 0219-1377, 0219-3116. doi:
10.1007/s10115-007-0114-2. url: http://link.springer.com/10.1007/s10115-007-0114-2.

[Xia+22] Shuyin Xia et al. “Ball kk-Means: Fast Adaptive Clustering With No Bounds”.
In: IEEE Transactions on Pattern Analysis and Machine Intelligence 44.1 (Jan.
2022), pp. 87–99. issn: 1939-3539. doi: 10.1109/TPAMI.2020.3008694. url:
https://ieeexplore.ieee.org/document/9139397.