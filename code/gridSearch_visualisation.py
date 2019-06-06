import pandas as pd
import json
import seaborn as sns
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.unicode']=True
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
from grid_search import senseEmbeddings


def gridSearchVisulisation(json_file = "../resources/gridSearch7.json" , save_path = "../report/img/grid.pdf"):
    '''function that visulazes the grid search
    args: json_file - data of hyperparameters and the correlation score
          save_path - destination path to save plot
    returns: subplots of swarmplots
    '''
    #load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    score = pd.DataFrame(data).sort_values(by='correlation:', ascending = False)
    score = score.drop(['epochs', 'negative'], axis=1)
    size = score.shape[1]
    score = score.rename(columns={'min_count': 'min count', 'correlation:': 'correlation'})
    q = np.concatenate((np.arange(size-2),np.arange(size-2)))
    d = score.drop(['correlation'],axis=1)

    f, axes = plt.subplots(size-2, size-1, figsize=(20,10), squeeze=True, sharey=True)

    for c_ind, column_to_drop in enumerate(list(d.columns)):
        data_ = d.drop(column_to_drop,axis=1)
        for index, val in enumerate(list(data_.columns)):
            a = sns.swarmplot(data = score,
                             x = column_to_drop,
                             y = 'correlation',
                           hue = val,
                            ax = axes[index, c_ind],
                             size = 10)
            if c_ind!=0:
                a.get_yaxis().set_visible(False)
            a.tick_params(labelsize=20)
            a.set_ylabel('correlation', fontsize=20)
            a.set_xlabel(column_to_drop, fontsize=20)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_path, bbox_inches = "tight")


def annotationsHistogram(file):
    '''creates a histogram of annorations per sentence'''

    x = np.loadtxt(file)
    plt.figure(figsize=(12,5))
    plt.hist(x, bins = 200)
    plt.xlabel("Number of annotations")
    plt.ylabel("Occurences")
    plt.title("Annotations in a sentence")
    plt.show()

if __name__ == '__main__':

    #plotting for report

    gridSearchVisulisation()

    annotationsHistogram(file = '../resources/parsed_corpora_annotations_final.txt')
