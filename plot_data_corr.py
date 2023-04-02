import pandas
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

excel_name = 'D:\\data\\s-2\\冰面湖面积与SAM相关性.xlsx'

df = pandas.read_excel(excel_name, )

df = df.drop(index=3)
print(df)

r, p = stats.pearsonr(df['面积'], df['SAM'])
print('相关系数r为 = %6.3f，p值为 = %6.3f' % (r, p))

plt.scatter(df['面积'], df['SAM'])
plt.show()