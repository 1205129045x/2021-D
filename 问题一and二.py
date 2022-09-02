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

# data=pd.concat((feature_train,label_train['pIC50']),axis=1)
# corrlation=data.corr()
# cor_1=pd.DataFrame(corrlation['pIC50'])
# cor_1=abs(cor_1)

# 灰色关联分析，筛选出200个变量

#应用随机森林来选择特征
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split

forest = RandomForestRegressor(n_estimators=500, random_state=0, max_features=100,  n_jobs=2) 
forest.fit(feature_train,label_train.values.ravel())

# 构建一个表格用来存储特征的重要性，定义rank是用来记住他的序号的
feature_impor=pd.DataFrame(forest.feature_importances_)
feature_impor['rank']=range(0,feature_impor.shape[0])
feature_impor.columns=['s','rank']
feature_impor=feature_impor.sort_values(by='s',ascending=False)

# 进行随机特征选择
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
# answer=['ALogP','ALogp2','AMR','C3SP2', 'C1SP2','ATSc1', 'ATSc2', 'ATSc3', 'ATSc4','LipoaffinityIndex','BCUTc-1l', 'BCUTc-1h', 'BCUTp-1l', 'BCUTp-1h','XLogP','MDEC-22', 'MDEC-23', 'MDEC-33','minssCH2', 'minHBa', 'mindssC','MLFER_A','CrippenLogP','nAcid']
# feature_train=pd.read_csv('Molecular_Descriptor.csv',index_col='SMILES')
final_feature_train=feature_train[choose_feature[:20]]

##############deep-森林##################
# 导入机器学习库
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from deepforest import CascadeForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
# 输入 X:final_feature_train 
#      Y:label_train.values.ravel

X=final_feature_train.values
y=label_train.values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=56)

###### 标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)

names = ["KNeighborsRegressor", "RBF SVR",
 "RandomForestRegressor", 
 "GradientBoostingRegressor",
 "Stacking",
 "CascadeForestRegressor"]

# estimators 存储4个集成学习里面的分类器
estimators = [('rf', RandomForestRegressor(n_jobs=-1)),('svr', SVR(kernel="rbf")),('gdbt',GradientBoostingRegressor()),('knn',KNeighborsRegressor(n_jobs=-1))]

classifiers = [
 KNeighborsRegressor(n_jobs=-1),
 SVR(kernel="rbf"),
 RandomForestRegressor(n_jobs=-1),
 GradientBoostingRegressor(),
 StackingRegressor(estimators=estimators,final_estimator=LinearRegression()),
 CascadeForestRegressor(verbose=False,n_jobs=-1,random_state=56),
]

regressor_df = pd.DataFrame(index = names,columns=['MSE','R2'])
predict_map = {}

for name, clf in zip(names, classifiers):
    print("---start\t" + name + "\t----")
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_valid)
    predict_map[name] = y_pred
    MSE = mean_absolute_error(y_pred,y_valid)
    R2 = r2_score(y_valid,y_pred)
    regressor_df.loc[name]['MSE']=MSE
    regressor_df.loc[name]['R2'] = R2

####### 第二问答案写入 #####
feature_test=pd.read_csv('Molecular_Descriptor_test.csv',index_col='SMILES')
feature_test=feature_test[choose_feature[:20]]
feature_test = scaler.transform(feature_test.values)
label_test=pd.read_csv('ERα_activity_test.csv',index_col='SMILES')

clf=CascadeForestRegressor(verbose=False,n_jobs=-1,random_state=56)
clf.fit(X_train, y_train.ravel())
y_pred = clf.predict(feature_test)
pd.DataFrame(y_pred).to_csv('第二问结果.csv')