from reval.best_nclust_cv import FindBestClustCV
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = make_blobs(1000, 3, centers=5, random_state=42)
plt.scatter(data[0][:, 0], data[0][:, 1],
            c=data[1], cmap='rainbow_r')
plt.show()

X_tr, X_ts, y_tr, y_ts = train_test_split(data[0], data[1],
                                          test_size=0.30,
                                          random_state=42,
                                          stratify=data[1])

classifier = KNeighborsClassifier()
clustering = AgglomerativeClustering()
findbestclust = FindBestClustCV(nfold=2,
                                nclust_range=list(range(2, 11)),
                                s=classifier,
                                c=clustering,
                                nrand=100,
                                n_jobs=10)
metrics, nbest = findbestclust.best_nclust(X_tr, iter_cv=10,) #strat_vect=y_tr)
out = findbestclust.evaluate(X_tr, X_ts, nbest)

from reval.visualization import plot_metrics
plot_metrics(metrics, title="Reval metrics")

import pandas as pd
import h5py
path_in = r'/home/bgeurten/ownCloud/data/WPuQh5pypd.hdf5' # get profile names 
f1 = h5py.File(path_in, 'r+') 
keys = list(f1['CONSUMER'].keys()) 
f1.close() 
del f1
df = [] 
for hh_name in keys: 
    try:
        data = pd.read_hdf('/home/bgeurten/ownCloud/data/WPuQh5pypd.hdf5', key=hh_name) 
        df.append(data) 
    except:
        print(hh_name)