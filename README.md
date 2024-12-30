## Index

0. [Index](#index)
1. [Standard Lloyd's Algorithm](#standard-lloyds-algorithm)
    1. [Pseudocode](#pseudocode)
    2. [Notes](#notes)
2. [Accuracy](#accuracy)
3. [Hartigan's Algorithm](#hartigans-algorithm)
    1. [Pseudocode (standard)](#pseudocode-standard)
    2. [Notes on extended algorithm](#notes-on-extended-algorithm)
    1. [Pseudocode (extended)](#pseudocode-extended)
    2. [Notes](#notes-1)
5. [Initialization](#initialization)
4. [Useful Resources](#useful-resources)


## Standard Lloyd's Algorithm

### Pseudocode

```
choose k = number of centroids
initialize centroids randomly

assign points to each centroid randomly
move the centroid to the mean of points assigned to it

repeat:
    reassign each datapoint to the closest centroid
    move the centroid to the mean of points assigned to it
until convergence
```

- Centroids $\mu_1,\dots,\mu_k$  are initialized uniformly at random on datapoints
- $c_i = \arg min_j ||x_i-\mu_j||$ for every $i$.
- $\mu_j = \frac{\sum_{i=1}^n \mathbb{1_{[c_i=j]}x_i}}{\sum_{i=1}^n \mathbb{1_{[c_i=j]}}}$


### Notes
- how to randomly choose init centroids: ok uniformly on data?
- kmeans++ for initialization?
- there will never be a centroid with no points assigned(?)
- May handle arrays. May be optimized using arrays.
- Default tolerance $10^{-6}$?

## Accuracy

To account for the fact that we can get a correct clustering permutating outputted clusters (i.e. to compare true and predicted labels without caring about the way the cluster is labelled) we solve a linear sum assignment problem.

The linear sum assignment problem is also known as minimum weight matching in bipartite graphs. A problem instance is described by a matrix $C$, where each $C[i,j]$ is the cost of matching vertex $i$ of the first partite set (a ‘worker’) and vertex $j$ of the second set (a ‘job’). The goal is to find a complete assignment of workers to jobs of minimal cost.

Formally, let $X$ be a boolean matrix where $X[i,j]=1$ iff row $i$ is assigned to column $j$. Then the optimal assignment has cost $$\min\sum_i\sum_j C_{i,j}X_{i,j}$$

where, in the case where the matrix $X$ is square, each row is assigned to exactly one column, and each column to exactly one row.

The problem is solved applying the Hungarian algorithm.

([source](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html))

## Hartigan's Algorithm

### Pseudocode (standard)
```
choose k = number of centroids

assign points to each centroid randomly
move the centroid to the mean of points assigned to it

repeat:
    for each datapoint d
        for each centroid c
            assign d to c
            compute the sum of squared distances from each point to c
        end for
        
        assign d to the centroid with smallest distance
        move the centroid to the mean of points assigned to it
    end for
until convergence
```

## Notes on extended algorithm
- Let $z$ points in $\R^d$.

- Let $c=\frac{1}{z}\sum_{i=1}^z$ their baricenter (centroid).

- The cost of the cluster is
$$cost = \sum_{i=1}^z||x_i-c||^2 = \sum_{i=1}^z||x_i||^2 -z||c||^2 $$

- By removing the first $s$ points ($s<z$). the barycenter of the removed set is $g = \frac{1}{s}\sum_{i=1}^s x_i$ and that of the remaining ones is $c' = \frac{zc-sg}{z-s}$.

- $$\Delta cost =-\left(\sum_{i=1}^s ||x_i||^2 - s||g||^2\right) - \frac{zs}{z-s}||c-g||^2$$

- Reassigning $s$ points from cluster 1 to cluster 2, we have
$$\Delta cost_{1\to2} = s\left(\frac{z_2}{z_2+s}||c_2 - g||^2 - \frac{z_1}{z_1+s}||c_1 - g||^2\right)$$

- Reassigning just one single point:
$$\Delta cost_{1\to2} = \frac{z_2}{z_2+1}||c_2 - x||^2 - \frac{z_1}{z_1+1}||c_1 - x||^2$$

- $\begin{cases}\Delta cost < 0 : \text{reassignment is convenient}\\ \Delta cost > 0 : \text{reassignment is not convenient}\end{cases}$

### Pseudocode (extended)

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



### Notes

- "One way of obtaining the initial cluster centres is suggested here. The points are
first ordered by their distances to the overall mean of the sample. Then, for cluster
L (L = 1,2, ..., K), the {1 + (L -1) * [M/K]}th point is chosen to be its initial cluster centre.
In effect, some K sample points are chosen as the initial cluster centres. Using this initialization
process, it is guaranteed that no cluster will be empty after the initial assignment in the
subroutine. A quick initialization, which is dependent on the input order of the points, takes
the first K points as the initial cent" (Hartigan, Wong)
- accept in order until the first that involves already modified clusters or accept until we made k/2 moves?
- why if unsafe mode fails revert all changes? can we sort and revert change by change?
- is safe mode always enabled to get standard Hartigan's?
- unsafe_mode is enabled again on every iteration of the loop?
- check random initialization for clusters

## Initialization
- random: select $k$ datapoints at random and take them as centroids
- random-data: assign to each datapoint one of $k$ centroids and calculate centroids afterwards
- kmeans++: first centroid is chosen randomly; the other are chosen with a certain probability among all datapoints, depending on the distance of them from the closest centroid.

### Notes
- probability of each datapoint to be a centroid is $$\frac{D(x)^2}{\sum_{x\in \mathcal X} D(x)^2}$$ where $D(x)$ is the shortest distance from a datapoint to the closest centroid.
- centroids in kmeans++ are initialized as zero list but to avoid one of the following behaviour only the first i's elements are considered in the distance calculation:
    - $[0., ... 0.]$ is the closest point to some datapoint, but is just a placeholder because the corresponding centroid not assigned yet
    - $[0., ... 0.]$ is actually a centroid, but it's ignored to avoid the precedent result.


## Useful resources

- [Hartigan's K-Means Versus Lloyd's K-Means - Is It Time for a Change?; Slonim, Aharoni, Crammer](https://www.ijcai.org/Proceedings/13/Papers/249.pdf)
- [Algorithm AS 136: A K-Means Clustering Algorithm; Hartigan, Wong
](https://doi.org/10.2307/2346830)
- [k-means++: The Advantages of Careful Seeding; Arthur, Vassilvitskii](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf)