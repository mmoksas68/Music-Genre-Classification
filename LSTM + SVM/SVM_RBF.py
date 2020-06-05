import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn import svm
from performance_metrics import report_performance

def extract_1d_data():
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    data = []
    label = []
    i = 0
    for g in genres:     
        for filename in os.listdir(f'Preprocess/rescaled_30_2/{g}'):
            songname = f'Preprocess/rescaled_30_2/{g}/{filename}'
            temp = plt.imread(songname)[:,:,:3]
            data.append(np.reshape(temp,(360*120*3)))
            label.append(i)
        i = i+1
    return data, label


# %% Extract and shuffle the data
X, y = extract_1d_data()

X, y = shuffle(X, y)

# %% Train the data for different parameters
train_X = X[:800]
test_X = X[800:]


train_y = y[:800]
test_y = y[800:]


for c in [100, 1000, 10000]:
    for g in [pow(2,-16), pow(2,-14), pow(2,-12), pow(2,-10)]:
        clf = svm.SVC(C=c, kernel='rbf', gamma=g)
        clf.fit(train_X, train_y)
        print("C:", c, "gamma:",g)
        predictions = clf.predict(test_X)
        report_performance(predictions, test_y)
        
#%%