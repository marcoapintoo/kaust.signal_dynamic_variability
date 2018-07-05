import numpy as np

class GenericProjector:
    """
    Provides the basic structure for
    projects a vector, matrix, or a sequence
    of both into another numerical space.
    """
    def __init__(self):
        pass
    
    def _project_vector(self, vector):
        raise NotImplementedError()
    
    def _project_matrix(self, matrix):
        return self._project_vector(matrix.ravel())

    def project(self, element):
        if len(element.shape) == 1:
            return self._project_vector(element)
        return self._project_matrix(element)

    def project_sequence(self, sequence):
        # Unoptimized
        return np.array([self.project(v) for v in sequence])

import sklearn.decomposition
class PCAProjector(GenericProjector):
    """
    Implements a LSH random projector.
    """
    def __init__(self, X, n_estimators=1):
        super().__init__()
        X = X.reshape(X.shape[0], -1)
        self.n_estimators = n_estimators
        self.pca = sklearn.decomposition.PCA(n_components=n_estimators, random_state=0).fit(X)
        print(":: PCA explained variance: {0}".format(self.pca.explained_variance_ratio_))

    def _project_vector(self, vector):
        vector = vector.reshape(1, -1)
        projection = self.pca.transform(vector).ravel()
        if self.n_estimators == 1:
            return projection[0]
        return projection
    
    def project_sequence(self, sequence):
        return super().project_sequence(sequence.reshape(sequence.shape[0], -1))

class RandomProjector(GenericProjector):
    """
    Implements a LSH random projector.
    """
    def __init__(self, n_estimators, random_state=0):
        super().__init__()
        """
        n_estimators: number of random vectors to be used. The hash range will be from 0 to 2 ** hash_length - 1.
        """
        self.n_estimators = n_estimators
        self.prng = np.random.RandomState(random_state)
        self.random_vector_count = n_estimators#int(np.log2(10 ** hash_length))
        self.random_vectors = None
    
    def _project_vector(self, vector):
        if self.random_vectors is None or self.random_vectors.shape[1] != len(vector):
            print(":: Recalculating the LSH/TVVAR random vectors of size {0}".format(len(vector)))
            self.random_vectors = self.prng.randn(self.random_vector_count, len(vector))
        # It could be optimized.
        # I prefered to maintaining in this way in order 
        # to let clear how is working the algorithm XD 
        hashing = 0
        projections = np.dot(self.random_vectors, vector)
        for v in projections:
            hashing = hashing << 1 # It is the same than: projection = projection * 2
            if v >= 0:
                hashing += 1
        return hashing
    
class OrderedRandomProjector(GenericProjector):
    """
    Implements a LSH random projector.
    """
    def __init__(self, n_estimators, random_state=0):
        super().__init__()
        """
        n_estimators: number of random vectors to be used. The hash range will be from 0 to 2 ** hash_length - 1.
        """
        self.n_estimators = n_estimators
        self.prng = np.random.RandomState(random_state)
        self.random_vector_count = n_estimators#int(np.log2(10 ** hash_length))
        self.random_vectors = None
    
    def _project_vector(self, vector):
        if self.random_vectors is None or self.random_vectors.shape[1] != len(vector):
            print(":: Recalculating the LSH/TVVAR random vectors of size {0}".format(len(vector)))
            self.random_vectors = np.sort(
                self.prng.exponential(size=(self.random_vector_count, len(vector)))
                - self.prng.exponential(size=(self.random_vector_count, len(vector)))
            )
            self.random_vectors = np.sort(self.prng.randn(self.random_vector_count, len(vector)) ** 3)
        # It could be optimized.
        # I prefered to maintaining in this way in order 
        # to let clear how is working the algorithm XD 
        hashing = 0
        projections = np.dot(self.random_vectors, vector)
        for v in projections:
            hashing = hashing << 1 # It is the same than: projection = projection * 2
            if v > 0:
                hashing += 1
        return hashing
    

if __name__ == "__main__":
    projector = RandomProjector(n_estimators=2, )
    vectors = np.array([
        [1, 0, 0, 2, 1, 10, 8],
        [1, 0, 0, 2, 1, 11, 8],
        [10, 0, 0, 2, 1, 10, 8],
        [1, 0, 0, 2, 1, 10, 8],
        [1, 1, -1, 2, 1, -11, 8],
    ])
    for vector in vectors:
        print("*", vector, projector.project(vector))
    print("=>", projector.project_sequence(vectors))

    sequence = np.array([
        [[1, 0], [0, 2], [1, 10], [8, -1] ],
        [[1, 0], [0, 2], [1, 11], [6, 2] ],
        [[10, 0], [0, 2], [1, 10], [-8, 0] ],
        [[1, 0], [0, 2], [1, 10], [8, -1] ],
        [[1, 1], [-1, 2], [1, -11], [8, 0] ],
    ])
    for vector in sequence:
        print("*", projector.project(vector), projector.project(vector.ravel()))
    print("=>", projector.project_sequence(sequence))

