import numpy as np
from collections import namedtuple

class MatrixPlotableObject:
    def __init__(self, hashcode, keynames, distance_matrix):
        self.hashcode = hashcode
        self.keynames = keynames
        self.distance_matrix = distance_matrix

class AsymptoticMatrixInverseProjection:
    def __init__(self, label_names, epsilon=1e-4):
        self.epsilon = epsilon
        self.dimensions = len(label_names)
        self.label_names = label_names
    
    def fit1(self, projector, min_counting=1000):
        print(":: Estimating asymptotic behaviors")
        M = 2 ** projector.n_estimators * 1000
        counting = {}
        vectors = {}
        squared_vectors = {}
        mean_epsilons = {}
        mean_epsilons_convergence = {}
        for k in range(M):
            #vector = np.random.normal(0, 1, size=(self.dimensions, self.dimensions))
            vector = np.random.normal(0, 0.6, size=(self.dimensions, self.dimensions))
            hashcode = projector.project(vector)
            counting.setdefault(hashcode, 0)
            vectors.setdefault(hashcode, 0)
            squared_vectors.setdefault(hashcode, 0)
            mean_epsilons.setdefault(hashcode, 0)

            counting[hashcode] += 1
            vectors[hashcode] += vector
            squared_vectors[hashcode] += vector ** 2
            """
            epsilons[hashcode] = np.mean((
                squared_vectors[hashcode].ravel() / counting[hashcode]
                -
                (vectors[hashcode].ravel() / counting[hashcode]) ** 2
            ))
            """
            mean_epsilon = np.sum(vectors[hashcode]/counting[hashcode])
            mean_epsilons_convergence[hashcode] = np.abs(mean_epsilons[hashcode] - mean_epsilon) < self.epsilon
            mean_epsilons[hashcode] = mean_epsilon
            #if min_counting < k and np.all(list(mean_epsilons_convergence)):
            #    print(":: Converged")
            #    break
        print(":: Hashcodes:", list(counting.keys()))
        self.convergence_vectors = {hashcode: vector/counting[hashcode] for hashcode, vector in vectors.items()}
        return [
            MatrixPlotableObject(
                hashcode=hashcode,
                keynames=self.label_names,
                distance_matrix=vector
            )
            for hashcode, vector in self.convergence_vectors.items()
        ]
        for hashcode, vector in vectors.items():
            print(hashcode, np.sum(
                vectors[hashcode]/counting[hashcode]
            ))
    
    def fit2(self, projector, min_counting=1000):
        print(":: Estimating asymptotic behaviors")
        M = 2 ** projector.n_estimators * 100000
        counting = {k: 0 for k in range(2 ** projector.n_estimators)}
        vectors = {k: 0 for k in range(2 ** projector.n_estimators)}
        for k in range(M):
            #vector = np.random.normal(0, 1, size=(self.dimensions, self.dimensions))
            vector = np.random.normal(0, 0.6, size=(self.dimensions, self.dimensions))
            hashcode = projector.project(vector)
            counting[hashcode] += 1
            vectors[hashcode] += vector
        print(":: Hashcodes:", list(counting.keys()))
        self.convergence_vectors = {hashcode: vector/counting[hashcode] for hashcode, vector in vectors.items()}
        return [
            MatrixPlotableObject(
                hashcode=hashcode,
                keynames=self.label_names,
                distance_matrix=vector
            )
            for hashcode, vector in self.convergence_vectors.items()
        ]
        for hashcode, vector in vectors.items():
            print(hashcode, np.sum(
                vectors[hashcode]/counting[hashcode]
            ))
    
    def fit(self, projector, min_counting=1000):
        #print(":: Estimating asymptotic behaviors", self.dimensions, self.dimensions)
        M = 2 ** projector.n_estimators
        projector.project(np.zeros((self.dimensions * self.dimensions)))
        self.convergence_vectors = {}
        for k in range(M):
            x = np.fromstring(np.binary_repr(k, width=projector.n_estimators), 'u1') - ord('0')
            # Zero when AX < 0, and One if AX > 0
            # Threfore, we need to transform this x: (0, 1) -> (-1, 1)
            x = 2 * x - 1
            #print(x, projector.random_vectors.shape)
            vector = np.dot(x, projector.random_vectors).reshape(self.dimensions, self.dimensions)
            # Rescaling
            print("rescaling")
            vector = vector / projector.n_estimators
            self.convergence_vectors[k] = vector 
        return [
            MatrixPlotableObject(
                hashcode=hashcode,
                keynames=self.label_names,
                distance_matrix=vector
            )
            for hashcode, vector in self.convergence_vectors.items()
        ]