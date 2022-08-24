import pandas as pd
# train_729 = pd.read_csv('Molecular_Descriptor.csv', index_col=[0])
feature_train=pd.read_csv('Molecular_Descriptor.csv',index_col='SMILES')

label_train=pd.read_csv('ERα_activity.csv',index_col='SMILES')
del label_train['IC50_nM'] # 删除这一列
data=pd.concat((feature_train,label_train['pIC50']),axis=1)
feature=feature_train.columns # 初始特征
# 删除数据只有一种的特征
for  s in feature_train.columns:
    if feature_train[s].nunique()==1:
        del feature_train[s]    
# 删除重复率大于0.9的特征
for  s in feature_train.columns:
    a=1974*0.9
    if feature_train[s].value_counts().iloc[0]>a:
        del feature_train[s]  

data=pd.concat((feature_train,label_train['pIC50']),axis=1)
corrlation=data.corr()
cor_1=pd.DataFrame(corrlation['pIC50'])
cor_1=abs(cor_1)

# 灰色关联分析，筛选出200个变量

#应用随机森林来选择特征
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

forest = RandomForestRegressor(n_estimators=500, random_state=0, max_features=100,  n_jobs=2) 
forest.fit(feature_train,label_train.values.ravel())

# 构建一个表格用来存储特征的重要性，定义rank是用来记住他的序号的
feature_impor=pd.DataFrame(forest.feature_importances_)
feature_impor['rank']=range(0,362)
feature_impor.columns=['s','rank']
feature_impor=feature_impor.sort_values(by='s',ascending=False)

# 进行随机特征选择
choose_feature=[]
for i in range(362):
    if feature_impor.iloc[i,0]>=float(feature_impor['s'].quantile(0.89)):
        choose_feature.append(feature_train.columns[feature_impor.iloc[i,1]])

# 然后通过相关分析提出自相关的变量
cor=feature_train[choose_feature].corr()
# 
for i in range(39):
    for  j in range(39):
        if i !=j:
            if cor.iloc[i,j]>0.5 and cor.index[i] in choose_feature and cor.columns[j] in choose_feature:
                if len(choose_feature)==20:
                    break;
                choose_feature.remove(cor.columns[j])