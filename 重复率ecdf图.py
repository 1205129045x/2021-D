import pandas as pd
data = pd.read_excel('ttt.xlsx')

# 画出重复率的函数密度图
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
rcParams.update(config) # 后更新配置，防止字体设置被覆盖
import matplotlib.font_manager
matplotlib.font_manager._rebuild()


sns.ecdfplot(data['重复率'])
# sns.set(style="white") #设置seaborn画图的背景为白色
# rcParams.update(config)
# plt.xlabel('pIC50',fontproperties='Times New Roman',size=18)
plt.ylabel('密度',size=18)
plt.xlabel('重复率',size=18)
# plt.title('$\mathrm{pIC50}$密度分布图')
plt.yticks(fontproperties='Times New Roman', size=18)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.title('重复率的累计分布图',size=18)
plt.savefig('重复率的ecdf图')
plt.show()

# plt.figure()
# plt.rc("font", size=14)
# sns.displot(data['pIC50'])
# # sns.set(style="white") #设置seaborn画图的背景为白色
# # sns.set(style="whitegrid", color_codes=True)
# plt.show()