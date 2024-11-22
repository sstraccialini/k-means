## Standard Lloyd's Algorithm

### Pseudocode

```
choose k = number of centroids

assign points to each centroid randomly
move the centroid to the mean of points associated to it

repeat:
    reassign each datapoint to the closest centroid
    move the centroid to the mean of points associated to it
until convergence
```

- Centroids $\mu_1,\dots,\mu_k$  are initialized uniformly at random on datapoints
- $c_i = \arg min_j ||x_i-\mu_j||$ for every $i$.
- $\mu_j = \frac{\sum_{i=1}^n \mathbb{1_{[c_i=j]}x_i}}{\sum_{i=1}^n \mathbb{1_{[c_i=j]}}}$


### Notes
- how to randomly choose init centroids: ok uniformly on data?
- is it useful kmeans++ for initialization?
- there will never be a centroid with no points associated (?)

## Accuracy

To account for the fact that we can get a correct clustering permutating outputted clusters (i.e. to compare true and predicted labels without caring about the way the cluster is labelled) we solve a linear sum assignment problem.

The linear sum assignment problem is also known as minimum weight matching in bipartite graphs. A problem instance is described by a matrix $C$, where each $C[i,j]$ is the cost of matching vertex $i$ of the first partite set (a ‘worker’) and vertex $j$ of the second set (a ‘job’). The goal is to find a complete assignment of workers to jobs of minimal cost.

Formally, let $X$ be a boolean matrix where $X[i,j]=1$ iff row $i$ is assigned to column $j$. Then the optimal assignment has cost $$\min\sum_i\sum_j C_{i,j}X_{i,j}$$

where, in the case where the matrix $X$ is square, each row is assigned to exactly one column, and each column to exactly one row.

The problem is solved applying the Hungarian algorithm.

([source](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html))
