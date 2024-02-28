import numpy as np
from scipy.stats import t as t_dist, norm, cauchy, laplace



class Benchmark:
    def __init__(self):
        pass

    def noiseless(self, X):
        raise NotImplementedError

    def quantile(self, X, q):
        raise NotImplementedError

    def sample(self, X):
        raise NotImplementedError

class Scenario1(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 1

    def noiseless(self, X):
        return np.exp(-1/(X[:,0]**6))
        

    def quantile(self, X, q):
        return self.noiseless(X) + 0.01*np.random.normal(size=X.shape[0])

    def sample(self, X):
        return self.noiseless(X) + 0.01*np.random.normal(size=X.shape[0])  


class Scenario2(Benchmark):
    def __init__(self):
        super().__init__()
        self.n_in = 3

    def noiseless(self, X):
        #return np.exp(-1/(X[:,0]**6))
        return 1.5*np.abs((X[:,2]-0.4)*(X[:,2]-0.6))+0.5*np.exp(X[:,1])+np.sin(2*np.pi*X[:,1])

    def quantile(self, X, q):
        return self.noiseless(X) + 0.1*X[:,1]*np.random.standard_t(3, size=X.shape[0])

    def sample(self, X):
        return self.noiseless(X) + 0.1*X[:,1]*np.random.standard_t(3, size=X.shape[0])











