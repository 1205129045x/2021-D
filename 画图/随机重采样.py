import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#### 构建admet的样本不平衡表格
ADMETlabel=pd.read_csv("Molecular_Descriptor_admet.csv")
ADMETlabel['powerlabel'] = ADMETlabel.apply(lambda x : 16*x["Caco-2"]+8*x['CYP3A4']+4*x['hERG']+2*x['HOB']+1*x['MN'],axis=1)
ADMETlabel['powerlabel'].hist(bins=np.unique(ADMETlabel['powerlabel']))


fig, ax = plt.subplots()
d_heights, d_bins = np.histogram(ADMETlabel['Caco-2'],bins=[-0.5,0.5,1.5])
# np.historgram返回的是横坐标和高度，即给定的bins上对应的高度，这里bins指的是区间，就对应数据的0和1
m_heights, m_bins = np.histogram(ADMETlabel['CYP3A4'], bins=d_bins)
s_heights, s_bins = np.histogram(ADMETlabel['hERG'], bins=m_bins)
ss_heights, ss_bins = np.histogram(ADMETlabel['HOB'], bins=s_bins)
t_heights, t_bins = np.histogram(ADMETlabel['MN'], bins=ss_bins)

width = (d_bins[1] - d_bins[0])/6.0
ax.bar(d_bins[:-1]+width, d_heights, width=width, facecolor='cornflowerblue',label='Caco-2')
# ax.bar_label(label_type='center')
ax.bar(m_bins[:-1]+width*2, m_heights, width=width, facecolor='seagreen',label='CYP3A4')
ax.bar(s_bins[:-1]+width*3, s_heights, width=width, facecolor='red',label='hERG')
ax.bar(ss_bins[:-1]+width*4, ss_heights, width=width, facecolor='blue',label='HOB')
ax.bar(t_bins[:-1]+width*5, t_heights, width=width, facecolor='yellow',label='MN')
ax.legend() # 添加图注
plt.show()

### 数据集的划分

train_df = pd.DataFrame(columns = ADMETlabel.columns)
val_df = pd.DataFrame(columns = ADMETlabel.columns)
train_inds, val_inds = train_test_split(np.array(list(range(ADMETlabel.shape[0]))),test_size=0.2,random_state=7)
train_df = ADMETlabel.iloc[train_inds,:].reset_index(drop=True)
val_df = ADMETlabel.iloc[val_inds,:].reset_index(drop=True)


#### Random Oversampling

powercount = {}
powerlabels = np.unique(train_df['powerlabel'])
for p in powerlabels:
    powercount[p] = np.count_nonzero(train_df['powerlabel']==p)
maxcount = np.max(list(powercount.values()))
for p in powerlabels:
    gapnum = maxcount - powercount[p]
    #print(gapnum)
    temp_df = train_df.iloc[np.random.choice(np.where(train_df['powerlabel']==p)[0],size=gapnum)]
    # 遍历每个powerlabel的值，在具有相同的powerlabel值的样本中随机选则与最多相同样本数量的值的差的样本
    train_df = train_df.append(temp_df,ignore_index=True)


fig, ax = plt.subplots()
train_df['powerlabel'].value_counts().plot.bar()


#### 随机重采样后再画图

fig, ax = plt.subplots()
d_heights, d_bins = np.histogram(train_df['Caco-2'],bins=[-0.5,0.5,1.5])
m_heights, m_bins = np.histogram(train_df['CYP3A4'], bins=d_bins)
s_heights, s_bins = np.histogram(train_df['hERG'], bins=m_bins)
ss_heights, ss_bins = np.histogram(train_df['HOB'], bins=s_bins)
t_heights, t_bins = np.histogram(train_df['MN'], bins=ss_bins)

width = (d_bins[1] - d_bins[0])/6.0
ax.bar(d_bins[:-1]+width, d_heights, width=width, facecolor='cornflowerblue',label='Caco-2')
ax.bar(m_bins[:-1]+width*2, m_heights, width=width, facecolor='seagreen',label='CYP3A4')
ax.bar(s_bins[:-1]+width*3, s_heights, width=width, facecolor='red',label='hERG')
ax.bar(ss_bins[:-1]+width*4, ss_heights, width=width, facecolor='blue',label='HOB')
ax.bar(t_bins[:-1]+width*5, t_heights, width=width, facecolor='yellow',label='MN')
ax.legend() # 添加图注
plt.show()
train_df.to_csv('随机重采样后的样本分布.csv')
