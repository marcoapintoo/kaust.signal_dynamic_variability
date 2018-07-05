import numpy as np
import random_projection
import tv_var

import matplotlib.pyplot as plt
def testing_different_levels():
    signal = np.concatenate([
        np.random.random((1000, 16)),
        np.random.random_integers(4, 12, (1000, 16)),
    ])
    tvprocess = tv_var.TimeVaryingVAR(step_size=50, window_type="hamming")
    signal = tvprocess.process(signal)
    
    for m, n_estimators in enumerate([2, 10, 800]):
        plt.subplot(3, 2, 2 * m + 1)
        projector1 = random_projection.RandomProjector(n_estimators=n_estimators, )
        plt.plot(projector1.project_sequence(signal))
        plt.title("RP with n_estimators={0}".format(n_estimators))

        plt.subplot(3, 2, 2 * m + 2)
        projector2 = random_projection.OrderedRandomProjector(n_estimators=n_estimators, )
        plt.plot(projector2.project_sequence(signal))
        plt.title("ORP with n_estimators={0}".format(n_estimators))
    plt.show()

def testing_different_seeds():
    signal = np.concatenate([
        np.random.random((1000, 16)),
        np.random.random_integers(4, 12, (1000, 16)),
    ])
    tvprocess = tv_var.TimeVaryingVAR(step_size=50, window_type="hamming")
    signal = tvprocess.process(signal)
    
    n_estimators = 1000
    for m, seed in enumerate([0, 20, 100, 1024, 12345]):
        plt.subplot(5, 2, 2 * m + 1)
        projector1 = random_projection.RandomProjector(n_estimators=n_estimators, random_state=seed)
        plt.plot(projector1.project_sequence(signal))
        plt.title("RP with seed={0}".format(seed))
        plt.subplot(5, 2, 2 * m + 2)
        projector1 = random_projection.OrderedRandomProjector(n_estimators=n_estimators, random_state=seed)
        plt.plot(projector1.project_sequence(signal))
        plt.title("ORP with seed={0}".format(seed))

    plt.show()

def testing_var():
    projector = random_projection.RandomProjector(n_estimators=4)
    signal = np.concatenate([
        np.random.random((1000, 16)),
        np.random.random_integers(4, 12, (1000, 16)),
    ])
    tvprocess = tv_var.TimeVaryingVAR(step_size=50, window_type="hamming")
    coefficients = tvprocess.process(signal)
    print(projector.project_sequence(coefficients))


if __name__ == "__main__":
    testing_var()
    testing_different_seeds()
    testing_different_levels()
