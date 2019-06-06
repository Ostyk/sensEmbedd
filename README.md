# senseEmbedd: Learning sense embeddings for word similarity
## for a detailed overview please refer to the [report](report/report.pdf)
![](report/img/tsne.gif)

![](report/img/tsne3D.png)

## Code structure inside [code](code) folder
- [preprocess.py](code/preprocess.py)
  - preprocessing of the Eurosense corpus
- [grid_search.py](code/grid_search.py)
  - Grid search of parameter for the best models, and also training of the best one
- [gridSearchVisulisation.py](code/preprocess.py)
  - visualization of the grid search performance and some other statistics
  - .py version of [gridSearch_visualisation.ipynb](code/gridSearch_visualisation.ipynb)
- [tsne.py](code/tsne.py)
  - TSNE visualization plots
  - .py version of [tsne.ipynb](code/tsne.ipynb)
