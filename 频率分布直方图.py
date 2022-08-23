import pandas as pd
data=pd.read_excel('ttt.xlsx')
# Matplotlib模块 
import matplotlib.pyplot as plt
##绘制直方图
plt.rcParams["font.sans-serif"]='SimHei'
plt.rcParams['axes.unicode_minus']=False
# %config InlineBackend.figure_format='svg' 这种命令是ipython里面用的，jupyter也可以，
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')           
plt.hist(x=data['重复率'],bins=40,
        color="steelblue",
        edgecolor="black")          

#添加x轴和y轴标签
plt.xlabel("数量")
plt.ylabel("重复率")

#添加标题
plt.title("重复率分布")

#显示图形
plt.show()                          