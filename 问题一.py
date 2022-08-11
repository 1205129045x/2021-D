# 导入文件，将excel保存为csv读取更加快
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
feature_1=feature_train.columns # 删除数据只有一种后留下的特征
zero_feature = list(set(list(feature)).difference(set(list(feature_1))))  # 求差集 在feature中但是不在feature_1中 这里就是指值全为0的变量





# 删除重复率大于0.9的特征
for  s in feature_train.columns:
    a=1974*0.9
    if feature_train[s].value_counts().iloc[0]>a:
        del feature_train[s]  
feature_2=feature_train.columns # 删除重复率大于0.9的特征后留下的特征

# 画出y的函数密度图
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import rcParams
# rcParams配置文件，用来定义各种变量的，这里用来定义字体
config = {
    "font.family":'serif',
    "font.size": 18,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
# plt.rcParams['font.family'] = ['SimSun','Times New Roman']   # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set_theme(style="dark", color_codes=True)
sns.set_theme(style="darkgrid", color_codes=True)
rcParams.update(config)
import matplotlib.font_manager
matplotlib.font_manager._rebuild()
sns.kdeplot(data['pIC50'])
# sns.set(style="white") #设置seaborn画图的背景为白色
# rcParams.update(config)
plt.xlabel('pIC50',fontproperties='Times New Roman',size=18)
plt.ylabel('密度',size=18)
plt.title('$\mathrm{pIC50}$密度分布图')
plt.savefig('Log_ROC')
plt.yticks(fontproperties='Times New Roman', size=18)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.show()
# plt.figure()
# plt.rc("font", size=14)
# sns.displot(data['pIC50'])
# # sns.set(style="white") #设置seaborn画图的背景为白色
# # sns.set(style="whitegrid", color_codes=True)
# plt.show()