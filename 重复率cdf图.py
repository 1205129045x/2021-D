import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_excel('ttt.xlsx')
import seaborn as sns
from matplotlib import rcParams
from matplotlib import rc_params_from_file
config=rc_params_from_file('C:/matplotlibrc',fail_on_error=True)
rcParams.update(config)
bins=np.arange(0,1.1,0.1)
# np.alltrue(cdf[0] == np.cumsum(hi[0])/float(image.size))
plt.xlabel("重复率")
plt.ylabel("数量")
plt.xticks(np.arange(0, 1.1, step=0.1)) # 定义x轴上的刻度
plt.title('重复率的cdf图')
plt.hist(data['重复率'], bins=bins,  histtype='step',stacked=False,cumulative=True)
plt.savefig('重复率的cdf图.png')
plt.show()