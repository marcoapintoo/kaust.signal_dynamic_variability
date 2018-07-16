import numpy as np

class EmpiricalConditionals:
    def __init__(self, bins):
        self.bins = bins
        self.histograms = {}
        self.distance_matrix = 0
    
    def divergence(self, x, y):
        """
        Symmetric Leibler-Kullback divergence
        """
        eps = 1e-100
        #return
        #return (np.sum(x * np.log10(eps + x / (eps + y))))
        (
            #np.abs
            (np.sum(x * np.log10(eps + x / (eps + y)))) + 
            #np.abs
            (np.sum(y * np.log10(eps + y / (eps + x))))
        )
        #return (np.sum(x * np.log10(eps + x / (eps + y))))
        return np.sqrt((
            np.abs(np.sum(y * np.log10(eps + y / (eps + x)))) *
            np.abs(np.sum(x * np.log10(eps + x / (eps + y))))
        ))

    @property
    def keynames(self):
        return list(self.histograms.keys())

    def fit(self, x, y, names=None):
        """
        x and y must be 1-D vectors
        """
        for y0 in np.unique(y):
            histogram_bins, _ = np.histogram(x[np.where(y == y0)[0]], bins=self.bins)
            key = y0 if names is None else names[y0]
            self.histograms[key] = histogram_bins
        self._create_distance_matrix()
    
    def _create_distance_matrix(self):
        self.distance_matrix = np.zeros((len(self.histograms), len(self.histograms)))
        keys = self.keynames
        for i in range(len(self.histograms)):
            for j in range(len(self.histograms)):
                self.distance_matrix[i][j] = self.divergence(
                    self.get_distribution(keys[i], normalized=True),
                    self.get_distribution(keys[j], normalized=True),
                )

    def get_distribution(self, key, normalized=True):
        if normalized:
            return self.histograms[key] / self.histograms[key].sum()
        return self.histograms[key]
