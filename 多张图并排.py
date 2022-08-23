import pandas as pd
# train_729 = pd.read_csv('Molecular_Descriptor.csv', index_col=[0])
feature_train=pd.read_csv('Molecular_Descriptor.csv',index_col='SMILES')
# import seaborn as sns
import matplotlib.pyplot as plt

# 删除数据只有一种的特征
for  s in feature_train.columns:
    if feature_train[s].nunique()==1:
        del feature_train[s]    
# 删除重复率大于0.9的特征
for  s in feature_train.columns:
    a=1974*0.9
    if feature_train[s].value_counts().iloc[0]>a:
        del feature_train[s]  
# feature_train.dtypes.value_counts() 统计变量类型有几个
# feature_train.dtypes[feature_train.dtypes=='int64'] 选出类型为整形的变量
# feature_train.dtypes[feature_train.dtypes=='int64'].sample(16) # 随机挑选16个
data=feature_train[feature_train.dtypes[feature_train.dtypes=='int64'].sample(16).index]
# plt.figure(figsize=(10, 10), dpi=80)
fig,axes=plt.subplots(4,4)
a=0

for i in range(4):
    for j in range(4):
        data.iloc[:,a].value_counts().plot.bar( y=data.columns[a],rot=0,ax=axes[i, j])
        a=a+1
# for i in range(1,17):
#     plt.subplot(4,4,i)
#         feature_train['ndsCH'].value_counts().plot.bar(rot=0)
#         plt.subplot(4,4,j)
#         ax[i, j].text(0.5, 0.5, str((i, j)),
#                       fontsize=18, ha='center')

# plt.bar(feature_train['ndsCH'],height=0.8)
plt.savefig('整形数据展示.png')
plt.show()