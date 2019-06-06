from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import get_tmpfile
from gensim.models.word2vec import LineSentence
import nltk
from tqdm import tqdm
import os
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import scipy
import numpy as np
import itertools
import pandas as pd
from sklearn.model_selection import ParameterGrid
import json

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

        os.makedirs(self.path_prefix, exist_ok=True)

    def on_epoch_end(self, model):
        savepath = get_tmpfile(
            '{}_epoch{}.model'.format(self.path_prefix, self.epoch)
        )
        model.save(savepath)
        print(
            "Epoch saved: {}".format(self.epoch + 1),
            "Starting next epoch"
        )
        self.epoch += 1

def senseEmbeddings(word, dictionary):
    '''returns the sense embeddings for a given word'''

    word = word.split()
    word = "_".join(word) if len(word)>1 else  word[0]
    senses = []

    for w in dictionary:
        wordC =  "_".join(w.split(":")[0].split("_")[:-1])
        if word.lower() == wordC.lower():
            senses.append(w)

    return senses

def wordSimilarity(w1, w2, dictionary, model):
    '''take two words and outputs a score of their similarity'''

    w1_senses = senseEmbeddings(w1, dictionary)
    w2_senses = senseEmbeddings(w2, dictionary)
    score = - 1.0

    if len(w1_senses)!=0 and len(w2_senses)!=0:
        combinations = itertools.product(w1_senses, w2_senses)
        for s1, s2 in combinations:
            score = max(score, model.wv.similarity(s1, s2))

    return score

class Word2VecModel(BaseEstimator):
    '''hyperparamter grid search'''
    def __init__(self, window=5, min_count=3, size=400, alpha=0.01, sample = 1e-5, negative=9, epochs = 15):
        self.w2v_model = None
        self.window = window
        self.min_count = min_count
        self.size = size
        self.alpha = alpha
        self.sample = sample
        self.epochs = epochs
        self.negative = negative

    def fit(self, data):
        '''model training'''
        # Initialize model
        self.w2v_model = Word2Vec(size=self.size,
                                  window=self.window,
                                  alpha=self.alpha,
                                  min_count = self.min_count,
                                  workers = 8,
                                  sample = self.sample,
                                  sg = 1)
        # Build vocabulary
        self.w2v_model.build_vocab(data)

        self.w2v_model.save("../resources/word2vec_final_lower_skip.model")

        # Train model
        self.w2v_model.train(sentences = data,
                             total_examples=self.w2v_model.corpus_count,
                             epochs=self.epochs,
                             callbacks = [EpochSaver("./checkpoints")])
        self.w2v_model.wv.save_word2vec_format('../resources/embeddings_final_lower_skip.vec', binary=False)
        return self

    def score(self, gold):
        '''scoring function based on word similarity resulting in a correlation score'''
        vocab = self.w2v_model.wv.vocab
        dictionary = list(vocab.keys())
        gold['cosine'] = gold.apply(lambda row: wordSimilarity(row['Word 1'],
                                                               row['Word 2'],
                                                               dictionary,
                                                               self.w2v_model),axis=1)

        correlation_2, p2 = scipy.stats.spearmanr(gold['Human (mean)'], gold['cosine'])
        return correlation_2

def GridSearch(hyperparameters, corpora):
    '''performs a grid search operation and returns the best hyperparamters based on a correlation score'''
    best_score = - 1
    gold = pd.read_csv('../data/combined.tab', delimiter = '\t')
    grid = ParameterGrid(hyperparameters)
    best_grid = None
    print("number of combinations: {}\n".format(len(list(grid))))

    grids = []
    for current_grid in tqdm(grid):
        #model
        current_model = Word2VecModel(**current_grid)
        current_model.fit(corpora)
        current_score = current_model.score(gold)
        del current_model

        #updating and writing
        current_grid.update({"correlation:":current_score})
        grids.append(current_grid)

        print(current_grid)

        if current_score > best_score:
            print("better: ", current_score)
            best_score = current_score
            best_grid = current_grid

    return best_grid, grids
hyperparameters = {'window' : [4, 5],
                   'min_count' : [1, 3],
                   'size' : [100, 300],
                   'negative':[13],
                   'alpha' : [0.09, 0.001],
                   'epochs': [50]}

final_hyperparameters = {'window' : [5],
                        'min_count' : [3],
                        'size' : [100],
                        'negative':[13],
                        'alpha' : [0.09],
                        'epochs': [50]}

if __name__ == '__main__':
    path_to_corpus = '../resources/parsed_corpora_final_lower.txt'

    #load
    sentences = LineSentence(path_to_corpus)

    #search and train
    best_grid, grids = GridSearch(final_hyperparameters, sentences)

    #saving, comment for final hyperparamters
    # with open("../resources/gridsearch7.json", 'w') as f1:
    #     for grid in grids:
    #         json.dump(grid, f1)

    print("GridSearch complete: ", best_grid)
