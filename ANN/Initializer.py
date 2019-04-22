import numpy as np

class He(object):
    '''
        W will be of the shape (n_out, n_in)
    '''
    def get_W(self, n_in, n_out):
        W = np.random.uniform(
                low = -np.sqrt(6. / (n_in )),
                high = np.sqrt(6. / (n_in )),
                size = (n_out, n_in)
        )
        return  W

    def get_b(self, n_out):
        return np.zeros((n_out, 1))

class Xavier(object):
    '''
        W will be of the shape (n_out, n_in)
    '''
    def get_W(self, n_in, n_out):
        W = np.random.uniform(
                low = -np.sqrt(6. / (n_in + n_out)),
                high = np.sqrt(6. / (n_in + n_out)),
                size = (n_out, n_in)
        )
        return  W

    def get_b(self, n_out):
        return np.zeros((n_out, 1))

class dumbInitializer(object):
    def get_W(self, n_in, n_out):
        W = np.random.uniform(
                low = -1,
                high = 1,
                size = (n_out, n_in)
        )
        return  W

    def get_b(self, n_out):
        return np.zeros((n_out, 1))
