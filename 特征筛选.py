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
