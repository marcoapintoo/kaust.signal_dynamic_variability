import numpy as np
import scipy.signal

class GenericTimeVaryingProcess:
    def __init__(self, window_type="boxcar", window_size=100, step_size=1):
        """
        window_type: boxcar, triang, blackman, hamming, hann, bartlett, flattop, parzen, bohman, blackmanharris, nuttall, barthann, kaiser (needs beta), gaussian (needs std), general_gaussian (needs power, width), slepian (needs width), chebwin (needs attenuation) exponential (needs decay scale), tukey (needs taper fraction)
        """
        self.window_size = window_size
        self.step_size = step_size
        self.window_type = window_type
    
    def process(self, signal):
        window = scipy.signal.get_window(self.window_type, self.window_size)
        windowed_signal = np.array([
            self._operation(window.reshape(-1, 1) * signal[i: i + self.window_size])
            for i in range(0, len(signal) - self.window_size, self.step_size)
        ])
        #windowed_data = window * sliding_window(dataset, size, stepsize=stepsize, padded=False, axis=0)
        return windowed_signal
    
    def _operation(self, signal):
        raise NotImplementedError()

class TimeVaryingAverage(GenericTimeVaryingProcess):
    def __init__(self, window_type="boxcar", window_size=100, step_size=1):
        super().__init__(window_type=window_type, window_size=window_size, step_size=step_size)
    
    def _operation(self, signal):
        return signal.mean()


def var_to_pdc(A):
    from scipy import linalg, fftpack
    #https://gist.github.com/agramfort/9875439
    p, N, N = A.shape
    n_fft = max(int(2 ** np.ceil(np.log2(p))), 512)
    A2 = np.zeros((n_fft, N, N))
    A2[1:p + 1, :, :] = A  # start at 1 !
    fA = fftpack.fft(A2, axis=0)
    freqs = fftpack.fftfreq(n_fft)
    I = np.eye(N)
    for i in range(n_fft):
        fA[i] = linalg.inv(I - fA[i])
    P = np.zeros((n_fft, N, N))
    sigma = np.ones(N)
    for i in range(n_fft):
        B = fA[i]
        B = linalg.inv(B)
        V = np.abs(np.dot(B.T.conj(), B * (1. / sigma[:, None])))
        V = np.diag(V)  # denominator squared
        P[i] = np.abs(B * (1. / np.sqrt(sigma))[None, :]) / np.sqrt(V)[None, :]
    return P.mean(axis=0).reshape(1, P.shape[1], P.shape[2])

class TimeVaryingVAR(GenericTimeVaryingProcess):
    def __init__(self, order=1, window_type="boxcar", window_size=100, step_size=1, engine="statsmodels"):
        super().__init__(window_type=window_type, window_size=window_size, step_size=step_size)
        self.order = order
        self._engine = None
        self.engine = engine
    
    @property
    def engine(self):
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        if engine == "statsmodels":
            self._engine = engine
            self._operation = self._operation_python
        elif engine == "R":
            self._engine = engine
            self._operation = self._operation_r
        else:
            raise ValueError("Invalid TVVAR engine: {0}".format(engine))

    def _operation_python(self, signal):
        import statsmodels.tsa.vector_ar.var_model as var_model
        model_fitted = var_model.VAR(signal).fit(maxlags=self.order, trend="ct")
        results = model_fitted.coefs
        if results.shape[0] < self.order:
            results = np.concatenate([
                model_fitted.coefs,
                np.zeros(((self.order - results.shape[0]), signal.shape[1], signal.shape[1]))
            ])
        if self.order == 1:
            results = results[0]
        return results

    def _operation_r(self, signal):
        import rpy2.robjects.packages
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        robjects.r('sink("/dev/null")')
        mts = robjects.packages.importr("MTS")
        results = np.array(mts.VAR(signal, p=self.order).rx2("Phi"), dtype="f4")
        robjects.r('sink()')
        if self.order == 1:
            results = results.reshape(signal.shape[1], signal.shape[1])
        else:
            results = results.reshape(-1, signal.shape[1], signal.shape[1])
        return results

def time_varying_coefficients(dataset, order_tvvar=1, size=100, stepsize=1):
    mts = import_r("MTS")
    order_tvvar = order_tvvar #Parameter
    size = size #Parameter
    stepsize = stepsize #Parameter
    windowed_data = sliding_window(dataset, size, stepsize=stepsize, padded=False, axis=0)
    tv_var_coefficients = []
    for window_index in range(0, windowed_data.shape[0]):
        slided_data = windowed_data[window_index, :, :].T
        var_estimated = np.array(silence_r(mts.VAR)(slided_data, p=order_tvvar).rx2("Phi"), dtype="f4")
        tv_var_coefficients.append(var_estimated)
        print(".", end="")
    return np.array(tv_var_coefficients)

if __name__ == "__main__":
    tvprocess = TimeVaryingAverage(step_size=100)
    signal = np.random.random(1000)
    print(tvprocess.process(signal))
    signal = np.random.random((1000, 2))
    print(tvprocess.process(signal))
    tvprocess = TimeVaryingVAR(step_size=100)
    #tvprocess.order = 2
    print(tvprocess.process(signal))
    tvprocess = TimeVaryingVAR(step_size=100)
    tvprocess.engine = "R"
    #tvprocess.order = 2
    print(tvprocess.process(signal))
