import numpy as np
from scipy import linalg
from collections import defaultdict
import re
import nltk

def tokenize_text(data):
    data = re.sub(r'[,!?;-]', '.', data)                                 #  Punktuations are replaced by .
    data = nltk.word_tokenize(data)                                     #  Tokenize string to words
    return [ ch.lower() for ch in data if ch.isalpha() or ch == '.']    #  Lower case and drop non-alphabetical tokens

def load_data(data_file):
    '''
    Inputs:
        data_file: training data file
    Outputs:
        data: tokenized words as a string array
        fdist: words NLTK frequency distribution
    '''
    with open('shakespeare.txt') as f:
        data = f.read()                                                 #  Read in the data
    
    data = tokenize_text(data)
    # Compute the frequency distribution of the words in the dataset (vocabulary)
    fdist = nltk.FreqDist(data)

    return data, fdist

def get_dict(data):
    """
    Input:
        data: the data you want to pull from
    Output:
        word2Ind: returns dictionary mapping the word to its index
        ind2Word: returns dictionary mapping the index to its word
    """

    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    # return these correctly
    word2Ind = {}
    ind2Word = {}
    for k in words:
        word2Ind[k] = idx
        ind2Word[idx] = k
        idx += 1
    return word2Ind, ind2Word

def relu(k):
    result = k.copy()
    result[result < 0] = 0
    return result
    
def sigmoid(z):
    # sigmoid function
    return 1.0/(1.0+np.exp(-z))

def softmax(z):
    '''
    Inputs: 
        z: output scores from the hidden layer
    Outputs: 
        yhat: prediction (estimate of y)
    '''
    
    # Calculate yhat (softmax)
    result = np.exp(z)
    return result / np.sum(result, axis=0)

def get_idx(words, word2Ind):
    '''
    Inputs:
        words: list of dictionary words
        word2Idx: words to index dictionary
    Outputs:
        idx: list of word indeces
    '''
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx


def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed


def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C):i] + data[(i+1):(i+C+1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq/num_ctx_words
        yield x, y
        i += 1
        if i >= len(data):
            print('i is being set to 0')
            i = 0


def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch = []


def compute_pca(data, n_components=2):
    """
    Input: 
        data: of dimension (m,n) where each row corresponds to a word vector
        n_components: Number of components you want to keep.
    Output: 
        X_reduced: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """

    m, n = data.shape

    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(R)
    # sort eigenvalue in decreasing order
    # this returns the corresponding indices of evals and evecs
    idx = np.argsort(evals)[::-1]

    evecs = evecs[:, idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :n_components]

    return np.dot(evecs.T, data.T).T

def load_embeds(fileName):
    embs = []
    with open(fileName, "rt") as f:
        for line in f:
            row = list(map(float, line.split(',')))
            if len(embs) == 0:
                embs = np.empty((0, len(row)))
            embs = np.vstack((embs, row))
    
    return embs

