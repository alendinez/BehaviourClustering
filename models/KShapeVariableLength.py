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

class KShapeVariableLength(KShape):
    '''
    Adapted from tslearn.clustering.KShape
    '''
    def fit(self, X, y=None):
        max_attempts = max(self.n_init, 10)

        self.labels_ = None
        self.inertia_ = numpy.inf
        self.cluster_centers_ = None

        self.norms_ = 0.
        self.norms_centroids_ = 0.

        self.n_iter_ = 0

        self._X_fit = X_
        self.norms_ = numpy.array([numpy.linalg.norm(x[~numpy.isnan(x)]) for x in X_]) # Calculation of the norm adapted for variable length

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
        self.norms_centroids_ = numpy.array([numpy.linalg.norm(x[~numpy.isnan(x)]) for x in self.cluster_centers_])
        self._post_fit(X_, best_correct_centroids, min_inertia)
        return self
        

    def _fit_one_init(self, X, rs):
        if hasattr(self.init, '__array__'):
            self.cluster_centers_ = self.init.copy()
        elif self.init == "random":
            indices = rs.choice(X.shape[0], self.n_clusters)
            self.cluster_centers_ = X[indices].copy()
        else:
            raise ValueError("Value %r for parameter 'init' is "
                             "invalid" % self.init)
        self.norms_centroids_ = numpy.array([numpy.linalg.norm(x[~numpy.isnan(x)]) for x in self.cluster_centers_])
        self._assign(X)
        old_inertia = numpy.inf

        for it in range(self.max_iter):
            old_cluster_centers = self.cluster_centers_.copy()
            self._update_centroids(X) # !!!!!!!!!!!!!!!!!!!!!!!
            self._assign(X)
            if self.verbose:
                print("%.3f" % self.inertia_, end=" --> ")

            if numpy.abs(old_inertia - self.inertia_) < self.tol or \
                    (old_inertia - self.inertia_ < 0):
                self.cluster_centers_ = old_cluster_centers
                self._assign(X)
                break

            old_inertia = self.inertia_
        if self.verbose:
            print("")

        self._iter = it + 1

        return self


    def _update_centroids(self, X):
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = self._shape_extraction(X, k)
        self.cluster_centers_ = TimeSeriesScalerMeanVariance(
            mu=0., std=1.).fit_transform(self.cluster_centers_)
        self.norms_centroids_ = numpy.array([numpy.linalg.norm(x[~numpy.isnan(x)]) for x in self.cluster_centers_])