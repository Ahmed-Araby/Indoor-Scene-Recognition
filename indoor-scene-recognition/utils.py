"""
some helper functions with general usage
"""
import pickle
import numpy as np

def save_into_dics(obj , name):
    # name should include the extension
    pickle_out = open(name , 'wb')
    pickle.dump(obj , pickle_out)
    pickle_out.close()
    return


def read_from_dics(name):
    pickle_in = open(name , 'rb')
    obj = pickle.load(pickle_in)
    pickle_in.close()
    return obj


def shuffle(X , Y):
    np.random.seed(1)
    index = np.random.permutation(len(X))

    X = X[index]
    Y = Y[index]

    return X , Y

