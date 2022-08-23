
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_excel('ttt.xlsx')
import seaborn as sns
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
sns.ecdfplot(data=data['重复率'],complementary=False)
plt.show()
