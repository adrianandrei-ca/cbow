from pathlib import Path
import json
import numpy as np

import nltk
from nltk.tokenize import word_tokenize

from utils import (
    softmax,
    relu,
    get_batches,
    compute_pca,
    get_dict,
    load_data,
    tokenize_text,
)

# Download sentence tokenizer
nltk.data.path.append(".")
nltk.download("punkt")

class CBoW:
    def __init__(self, half_coverage=2, embed_size=50, num_iters=250):
        self.C = half_coverage
        self.N = embed_size
        self.num_iters = num_iters

        self.word2Ind = {}  # word token index map
        self.ind2word = {}  # index to word token map
        self.V = 0  # vocabulary size

        self.W1 = None  # N x V weight matrix to hidden layer
        self.W2 = None  # V x N hidden layer weight matrix
        self.b1 = None  # N bias weight array
        self.b2 = None  # V bias weight array

        self.embs = None  # V x N embeddings

        self.data = []

    def initialize_model(self, random_seed=1):
        """
        Inputs:
            N:  dimension of hidden vector
            V:  dimension of vocabulary
            random_seed: random seed for consistent results in the unit tests
        Outputs:
            W1, W2, b1, b2: initialized weights and biases
        """

        np.random.seed(random_seed)

        # W1 has shape (N,V)
        self.W1 = np.random.rand(self.N, self.V)
        # W2 has shape (V,N)
        self.W2 = np.random.rand(self.V, self.N)
        # b1 has shape (N,1)
        self.b1 = np.random.rand(self.N, 1)
        # b2 has shape (V,1)
        self.b2 = np.random.rand(self.V, 1)

        return self.W1, self.W2, self.b1, self.b2

    def forward_prop(self, x):
        """
        Inputs:
            x:  average one hot vector for the context
            W1, W2, b1, b2:  matrices and biases to be learned
        Outputs:
            z:  output score vector
        """

        # Calculate h
        h = np.dot(self.W1, x) + self.b1

        # Apply the relu on h (store result in h)
        h = np.maximum(0, h)

        # Calculate z
        z = np.dot(self.W2, h) + self.b2

        return z, h

    # compute_ cost as a cross-entropy cost function
    def compute_cost(self, y, yhat, batch_size):
        # Calculate element-wise log probabilities for both true (y) and false (1 - y) cases
        # This implements part of the cross-entropy formula: y * log(y_hat) + (1 - y) * log(1 - y_hat)
        logprobs  = np. multiply(np.log(yhat), y)  + np.multiply(np.log(1  - yhat), 1  - y)
        
        # Compute the average cost over the batch by summing and dividing by the batch size
        # The negative sign is applied to match the standard cross-entropy loss definition
        cost  =  -1 / batch_size * np.sum(logprobs)
        
        # Squeeze out any extra dimensions from the cost array to return a scalar value
        cost  = np.squeeze(cost)
        
        return cost

    def back_prop(self, x, yhat, y, h, batch_size):
        """
        Inputs:
            x:  average one hot vector for the context
            yhat: prediction (estimate of y)
            y:  target vector
            h:  hidden vector (see eq. 1)
            batch_size: batch size
        Outputs:
            grad_W1, grad_W2, grad_b1, grad_b2:  gradients of matrices and biases
        """

        grad_W1 = (1 / batch_size) * np.dot(relu(np.dot(self.W2.T, yhat - y)), x.T)
        grad_W2 = (1 / batch_size) * np.dot(yhat - y, h.T)
        grad_b1 = ((1 / batch_size) * relu(np.dot(self.W2.T, yhat - y))).sum(
            axis=1, keepdims=True
        )
        grad_b2 = ((1 / batch_size) * (yhat - y)).sum(axis=1, keepdims=True)

        return grad_W1, grad_W2, grad_b1, grad_b2

    def gradient_descent(self, data, alpha=0.03):
        """
        This is the gradient_descent function

        Inputs:
            data:      text as an array of dictionary words
            word2Ind:  words to indices
            N:         dimension of hidden vector
            V:         dimension of vocabulary
            num_iters: number of iterations
        Outputs:
            W1, W2, b1, b2:  updated matrices and biases

        """
        self.initialize_model(random_seed=282)
        batch_size = 128
        iters = 0
        for x, y in get_batches(data, self.word2Ind, self.V, self.C, batch_size):
            # Get z and h
            z, h = self.forward_prop(x)
            # Get yhat
            yhat = softmax(z)
            # Get cost
            cost = self.compute_cost(y, yhat, batch_size)
            if (iters + 1) % 10 == 0:
                print(f"iters: {iters + 1} cost: {cost:.6f}")
            # Get gradients
            grad_W1, grad_W2, grad_b1, grad_b2 = self.back_prop(
                x, yhat, y, h, batch_size
            )

            # Update weights and biases
            self.W1 -= alpha * grad_W1
            self.W2 -= alpha * grad_W2
            self.b1 -= alpha * grad_b1
            self.b2 -= alpha * grad_b2

            iters += 1
            if iters == self.num_iters:
                print(f"Final cost: {cost:.6f}")
                break
            if iters % 100 == 0:
                alpha *= 0.66

        return self.W1, self.W2, self.b1, self.b2

    def sentence_embs(self, word_list):
        """
        Inputs:
            word_list: tokenized sentence
            embs: embedding matrix
            word2Ind: token word to index map
        Outputs:
            sentence embeddings
        """
        sentence_idx = [
            self.word2Ind[word] for word in word_list if word in self.word2Ind
        ]
        sentence_X = self.embs[sentence_idx, :]
        return np.mean(sentence_X, axis=0)

    def text_embs(self, sentence_array):
        text_embs_list = []
        for sentence in sentence_array:
            if sentence:
                word_list = tokenize_text(sentence)
                text_embs_list.append(self.sentence_embs(word_list))
        return np.vstack(text_embs_list)

    def random_embs(self, rows):
        cols = self.embs.shape[1]
        return np.random.uniform(0, 1, (rows, cols))

    def load_training_data(self, fileName="shakespeare.txt"):
        self.data, self.fdist = load_data(fileName)

        # get_dict creates two dictionaries, converting words to indices and viceversa.
        self.word2Ind, self.ind2Word = get_dict(self.data)
        self.V = len(self.word2Ind)

    def train_model(self):
        print("Calling gradient_descent")
        self.gradient_descent(self.data)
        # NxN embedding matrix as a weights average
        self.embs = (self.W1.T + self.W2) / 2.0

    def slack_distance(self, embs, pca, slack=0.01):
        sentence_count = embs.shape[0]
        embs_dist = np.zeros((sentence_count, sentence_count))
        for i in range(1, sentence_count):
            for j in range(i):
                embs_dist[i, j] = np.linalg.norm(embs[i] - embs[j])

        pca_dist = np.zeros((sentence_count, sentence_count))
        for i in range(1, sentence_count):
            for j in range(i):
                pca_dist[i, j] = np.linalg.norm(pca[i] - pca[j])

        count = 0
        dist = 0
        for i in range(1, sentence_count):
            embs_row = embs_dist[i, 0:i]
            embs_arg = np.argsort(embs_row / np.sum(embs_row))
            pca_row = pca_dist[i, 0:i]
            pca_args = np.argsort(pca_row / np.sum(pca_row))

            for j in range(i):
                embs_diff = np.abs(
                    embs_dist[i, embs_arg[j]] - embs_dist[i, pca_args[j]]
                )
                pca_diff = np.abs(pca_dist[i, embs_arg[j]] - pca_dist[i, pca_args[j]])

                diff = np.max((embs_diff, pca_diff))
                if embs_arg[j] != pca_args[j] and diff > slack:
                    count += 1
                    dist = np.max((dist, diff))
        print(f"difference count: {count}")
        return count, dist

    def sentence_pca(self, sentences, dimensions=3, slack=-1):
        """
        Inputs:
            sentences - sentence array
            dimensions - the size of the principal component analysis array
        Outputs:
            N x dimensions PCA matrix
        """
        Y = self.text_embs(sentences)
        sentences_pca = compute_pca(Y, dimensions)
        if slack >= 0:
            print(self.slack_distance(Y, sentences_pca, slack))
        return sentences_pca

    def save_embeddings(self, folderName):
        path = Path(folderName)
        path.mkdir(parents=True, exist_ok=True)

        np.savetxt(path / "embs.txt", self.embs)
        with open(path / "word2Ind.json", "w") as f:
            json.dump(self.word2Ind, f)

    def load_embeddings(self, folderName):
        path = Path(folderName)

        if not (path / "embs.txt").exists():
            print(f"Embeddings not found in folder {folderName}")
            return False
        if not (path / "word2Ind.json").exists():
            print(f"Word index file not found in folder {folderName}")
            return False

        self.embs = np.loadtxt(path / "embs.txt")
        with open(path / "word2Ind.json", "r") as f:
            self.word2Ind = json.load(f)

        return True
