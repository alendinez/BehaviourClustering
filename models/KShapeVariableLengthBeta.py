from sklearn.utils import check_random_state
import numpy

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset, check_dims
from tslearn.clustering import KShape

from tslearn.clustering.utils import (TimeSeriesCentroidBasedClusteringMixin,
                    _check_no_empty_cluster, _compute_inertia,
                    _check_initial_guess, EmptyClusterError)

from tslearn.metrics import cdist_normalized_cc, y_shifted_sbd_vec

class KShapeVariableLengthBeta(KShape):
    '''
    Adapted from tslearn.clustering.KShape
    '''
    def fit(self, X, y=None, cross_dists=None):
        '''
        Parameters
        ----------
        X : array-like of shape=(n_ts, sz, d)
            Time series dataset.

        cross_dists: array-like of shape=(n_ts, n_ts)
            Maximum normalized cross-correlation between each pair of segments
        '''
        max_attempts = max(self.n_init, 10)

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None

        self.norms_ = 0.
        self.norms_centroids_ = 0.

        self.n_iter_ = 0

        X_ = to_time_series_dataset(X)
        self._X_fit = X_

        if cross_dists is None:
            self.norms_ = numpy.linalg.norm(X_, axis=(1, 2))
            self.cross_dists = cdist_normalized_cc(X_, X_,
                                        norms1=self.norms_,
                                        norms2=self.norms_,
                                        self_similarity=False)
        else:
            self.cross_dists = cross_dists

        _check_initial_guess(self.init, self.n_clusters)

        rs = check_random_state(self.random_state)

        best_correct_centroids = None
        min_inertia = numpy.inf
        n_successful = 0
        n_attempts = 0
        while n_successful < self.n_init and n_attempts < max_attempts:
            try:
                if self.verbose and self.n_init > 1:
                    print("Init %d" % (n_successful + 1))
                n_attempts += 1
                self._fit_one_init(X_, rs)
                if self.inertia_ < min_inertia:
                    best_correct_centroids = self.cluster_centers_.copy()
                    min_inertia = self.inertia_
                    self.n_iter_ = self._iter
                n_successful += 1
            except EmptyClusterError:
                if self.verbose:
                    print("Resumed because of empty cluster")
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self
        

    def _fit_one_init(self, X, rs):
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif self.init == "random":
            self.cluster_centers_ = rs.choice(X.shape[0], self.n_clusters)
        else:
            raise ValueError("Value %r for parameter 'init' is "
                             "invalid" % self.init)
        self._assign(X)
        old_inertia = numpy.inf
        old_labels = self.labels_

        for it in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy()
            self._update_centroids(X)
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if numpy.abs(old_inertia - self.inertia_) < self.tol or \
                    (old_labels == self.labels_).all():
                if old_inertia > self.inertia_:
                    self.cluster_centers_ = old_cluster_centers
                    self._assign(X)
                break

            old_inertia = self.inertia_
            old_labels = self.labels_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self


    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = self._centroid_selection(k)

    def _assign(self, X):
        dists = 1. - self.cross_dists
        self.labels_ = dists[:,self.cluster_centers_].argmin(axis=1)
        _check_no_empty_cluster(self.labels_, self.n_clusters)
        self.inertia_ = _compute_inertia(dists[:,self.cluster_centers_], self.labels_)

    def _centroid_selection(self, k):
        # Select submatrix of cross dists with label k
        idxs = (self.labels_ == k).nonzero()[0]
        dists = 1 - self.cross_dists[numpy.ix_(idxs,idxs)]
        # Get the one with max value
        centroid = idxs[dists.sum(axis=1).argmax()]
        return centroid
        

    