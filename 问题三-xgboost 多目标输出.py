import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.multioutput import MultiOutputClassifier

feature_train=pd.read_csv('Molecular_Descriptor.csv',index_col='SMILES')
label_train=pd.read_csv('ADMET.csv',index_col='SMILES')
test_x=pd.read_csv('Molecular_Descriptor_test.csv',index_col='SMILES')
# 删除数据只有一种的特征
for  s in feature_train.columns:
    if feature_train[s].nunique()==1:
        del feature_train[s]    

# 删除重复率大于0.9的特征
for  s in feature_train.columns:
    a=1974*0.9
    if feature_train[s].value_counts().iloc[0]>a:
        del feature_train[s]  

### SMOTE过采样




clf = xgb.XGBClassifier(tree_method="hist")
X=feature_train.values
y=label_train.values
clf.fit(X, y)
# result=pd.DataFrame(clf.predict(test_x))
# result.to_csv('第三问xgboost结果.csv')
# print(clf.predict(test_x))
# np.testing.assert_allclose(clf.predict(X), y)
#### 特征重要性计算
# 构建一个表格用来存储特征的重要性，定义rank是用来记住他的序号的
feature_impor=pd.DataFrame(clf.feature_importances_)
feature_impor['rank']=range(0,feature_impor.shape[0])
feature_impor.columns=['s','rank']
feature_impor=feature_impor.sort_values(by='s',ascending=False)
choose_feature=[]
for i in range(20):
    choose_feature.append(feature_train.columns[feature_impor.iloc[i,1]])


#### xgboost计算第一问筛选变量
