import argparse
from matplotlib import pyplot
import numpy as np

from cbow import CBoW
from utils import compute_pca, load_embeds

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='AI LangChain chatbot sample')

# Add arguments
parser.add_argument('-m', '--model', default = 'embs', help='Model folder path')
parser.add_argument('-s', '--sentences', required = False, help='Sentence file')
parser.add_argument('-e', '--embeddings', required = False, help='User embeddings text file, one per row')
parser.add_argument('-o', '--output', default='pca.svg', help='Output PCA SVG image file name')
parser.add_argument('-t', '--test', required = False, help='Run the tests', action="store_true")
parser.add_argument('-l', '--load', required = False, help='Load the model from model folder path', action="store_true")
args = parser.parse_args()

model = CBoW()
if args.load:
    model.load_embeddings(args.model)
else:
    model.load_training_data()
    model.train_model()
    model.save_embeddings(args.model)

def plot_pca_data(result):
    # plot the PCA data in 3d
    fig = pyplot.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(result[:, 0], result[:, 1], result[:, 2])
    for i in range(result.shape[0]):
        ax.text(result[i, 0], result[i, 1], result[i, 2], str(i + 1))

    pyplot.savefig(args.output)

def run_tests():
    sentences = [
        "Long live the king.",
        "To be or not to be!",
        "Cruel to be kind",
        "The clothes make the man",
        "In my heart of hearts",
        "A dish fit for the Gods",
        "We few, we happy few, we band of brothers",
        "Age cannot wither her, nor custom stale",
        "You wrong this presence; therefore speak no more"
    ]

    # compute the 3 dimensional PCAs
    result = model.sentence_pca(sentences, slack = 0.08)
    plot_pca_data(result)

def load_sentences(filename):
    with open(filename) as f:
        lines = f.read()
        lines = lines.split('\n')

    # compute the 3 dimensional PCAs
    result = model.sentence_pca(lines, slack = 0.08)
    plot_pca_data(result)

def plot_user_embs(fileName):
    Y = load_embeds(fileName)
    user_pca = compute_pca(Y, 3)
    print(model.slack_distance(Y, user_pca, slack= 0.08))
    plot_pca_data(user_pca)

if __name__ == "__main__":
    if args.test:
        run_tests()
    if args.sentences:
        load_sentences(args.sentences)
    if args.embeddings:
        plot_user_embs(args.embeddings)