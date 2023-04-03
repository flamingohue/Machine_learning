'''
This is a learning material from STA414 from University of Toronto. Using naive Bayes to predict new images.
'''
import numpy as np
import os
import gzip
import struct
import array
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


class NaiveBayes():
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

    def train_map_estimator(self):
        """ Inputs: train_images (N_samples x N_features), train_labels (N_samples x N_classes)
            Returns the MAP estimator theta_est (N_features x N_classes) and the MLE
            estimator pi_est (N_classes)"""
        
        # YOU NEED TO WRITE THIS PART
        self.pi_est = np.sum(self.train_labels, 0) / self.train_labels.shape[0] 
        # sum all over N_samples and follow the formula above
        
        self.theta_est = (1 + np.matmul(self.train_images.T, self.train_labels)) / (2 + np.sum(self.train_labels, 0))
        

    def log_likelihood(self, images):
        """ Inputs: images (N_samples x N_features), theta, pi
            Returns the matrix 'log_like' of loglikehoods over the input images where
            log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
            log_like is a matrix of (N_samples x N_classes)
        Note that log likelihood is not only for c^(i), it is for all possible c's."""

        # YOU NEED TO WRITE THIS PART
        log_p_x_given_c_theta = np.matmul(images, np.log(self.theta_est)) + \
        np.matmul(1 - images, np.log(1-self.theta_est))
        log_p_x_c = np.log(self.pi_est) + log_p_x_given_c_theta # numerator
        p_x_given_c_theta = np.exp(log_p_x_given_c_theta)

        log_like = log_p_x_c - \
        np.log(np.matmul(p_x_given_c_theta, self.pi_est)).reshape(-1, 1)
        # reshape is to make sure the shape is N_samples x N_classes
        return log_like

    def accuracy(self, log_like, labels):
        """ Inputs: matrix of log likelihoods and 1-of-K labels (N_samples x N_classes)
        Returns the accuracy based on predictions from log likelihood values"""

        # YOU NEED TO WRITE THIS PART
        accuracy = np.mean(np.argmax(log_like, 1) == np.argmax(labels, 1))
        return accuracy

                    

    




if __name__=='__main__':
    #load image data
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    model = NaiveBayes(train_images,train_labels)
    model.train_map_estimator()

    loglike_train = model.log_likelihood(train_images)
    avg_loglike = np.sum(loglike_train * train_labels) / N_data
    train_accuracy = model.accuracy(loglike_train, train_labels)
    loglike_test = model.log_likelihood(test_images)
    test_accuracy = model.accuracy(loglike_test, test_labels)
    print(f"Average log-likelihood for MAP is {avg_loglike:.3f}")
    print(f"Training accuracy for MAP is {train_accuracy:.3f}")
    print(f"Test accuracy for MAP is {test_accuracy:.3f}")


