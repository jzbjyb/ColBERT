from faiss import ClusteringParameters, Clustering, copy_array_to_vector, IndexFlatIP, IndexFlatL2, \
    index_cpu_to_gpus_list, index_cpu_to_all_gpus, ProgressiveDimClustering, GpuProgressiveDimIndexFactory, \
    ProgressiveDimIndexFactory, vector_float_to_array, swig_ptr
import numpy as np

def faiss_kmeans_train(self, x, weights=None, init_centroids=None):
    """ Perform k-means clustering.
    On output of the function call:
    - the centroids are in the centroids field of size (`k`, `d`).
    - the objective value at each iteration is in the array obj (size `niter`)
    - detailed optimization statistics are in the array iteration_stats.
    Parameters
    ----------
    x : array_like
        Training vectors, shape (n, d), `dtype` must be float32 and n should
        be larger than the number of clusters `k`.
    weights : array_like
        weight associated to each vector, shape `n`
    init_centroids : array_like
        initial set of centroids, shape (n, d)
    Returns
    -------
    final_obj: float
        final optimization objective
    """
    n, d = x.shape
    assert d == self.d

    if self.cp.__class__ == ClusteringParameters:
        # regular clustering
        clus = Clustering(d, self.k, self.cp)
        if init_centroids is not None:
            nc, d2 = init_centroids.shape
            assert d2 == d
            copy_array_to_vector(init_centroids.ravel(), clus.centroids)
        if self.cp.spherical:
            self.index = IndexFlatIP(d)
        else:
            self.index = IndexFlatL2(d)
        if type(self.gpu) is list:
            self.index = index_cpu_to_gpus_list(self.index, gpus=self.gpu)
        elif self.gpu:
            self.index = index_cpu_to_all_gpus(self.index, ngpu=self.gpu)
        clus.train(x, self.index, weights)
    else:
        # not supported for progressive dim
        assert weights is None
        assert init_centroids is None
        assert not self.cp.spherical
        clus = ProgressiveDimClustering(d, self.k, self.cp)
        if self.gpu:
            fac = GpuProgressiveDimIndexFactory(ngpu=self.gpu)
        else:
            fac = ProgressiveDimIndexFactory()
        clus.train(n, swig_ptr(x), fac)

    centroids = vector_float_to_array(clus.centroids)

    self.centroids = centroids.reshape(self.k, d)
    stats = clus.iteration_stats
    stats = [stats.at(i) for i in range(stats.size())]
    self.obj = np.array([st.obj for st in stats])
    # copy all the iteration_stats objects to a python array
    stat_fields = 'obj time time_search imbalance_factor nsplit'.split()
    self.iteration_stats = [
        {field: getattr(st, field) for field in stat_fields}
        for st in stats
    ]
    return self.obj[-1] if self.obj.size > 0 else 0.0
