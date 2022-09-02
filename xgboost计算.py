import pandas as pd
import xgboost as xgb

feature_train=pd.read_csv('Molecular_Descriptor.csv',index_col='SMILES')
label_train=pd.read_csv('ERα_activity.csv',index_col='SMILES')
del label_train['IC50_nM'] # 删除这一列
# 删除数据只有一种的特征
for  s in feature_train.columns:
    if feature_train[s].nunique()==1:
        del feature_train[s]    

# 删除重复率大于0.9的特征
for  s in feature_train.columns:
    a=1974*0.9
    if feature_train[s].value_counts().iloc[0]>a:
        del feature_train[s]  

clf = xgb.XGBRegressor(tree_method="hist")
X=feature_train.values
y=label_train.values
clf.fit(X, y)

# 构建一个表格用来存储特征的重要性，定义rank是用来记住他的序号的
feature_impor=pd.DataFrame(clf.feature_importances_)
feature_impor['rank']=range(0,feature_impor.shape[0])
feature_impor.columns=['s','rank']
feature_impor=feature_impor.sort_values(by='s',ascending=False)
choose_feature=[]
for i in range(feature_impor.shape[0]):
    if feature_impor.iloc[i,0]>=float(feature_impor['s'].quantile(0.89)):
        choose_feature.append(feature_train.columns[feature_impor.iloc[i,1]])

# 然后通过相关分析提出自相关的变量
cor=feature_train[choose_feature].corr()
# 
for i in range(len(cor)):
    for  j in range(len(cor)):
        if i !=j:
            if cor.iloc[i,j]>0.9 and cor.index[i] in choose_feature and cor.columns[j] in choose_feature:
                if len(choose_feature)==20:
                    break;
                choose_feature.remove(cor.columns[j])

final_feature_train=feature_train[choose_feature[:20]]
