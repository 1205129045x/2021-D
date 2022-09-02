from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import BorderlineSMOTE
import pandas as pd

feature_train=pd.read_csv('Molecular_Descriptor.csv',index_col='SMILES')
label_train=pd.read_csv('ADMET.csv',index_col='SMILES')
feature_train=feature_train.values
label_train=label_train.values
sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
X_res, y_res = sm.fit_resample(feature_train, label_train)

X, y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.1, 0.9], n_informative=2, n_redundant=0, flip_y=0,
                           n_features=2, n_clusters_per_class=1, n_samples=100, random_state=9)
print('Original dataset shape %s' % Counter(y))
sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
