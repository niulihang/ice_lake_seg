import pandas
import matplotlib.pyplot as plt
import math

excel_name = 'D:\\data\\s-2\\2017-2022Amery冰面湖面积.xlsx'

df = pandas.read_excel(excel_name, )
df['water面积'] = round(df['water面积'] / math.pow(10, 6), 2)
print(df)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
fig = plt.figure(figsize=(12, 8))
plt.grid(axis='both', linestyle='-.')
plt.xticks(fontsize=15)
plt.title('Amery冰架2017-2022年1月份表面融水面积变化图', fontsize=20)

# ax1 = fig.add_subplot(111)
# ax1.bar(df['年份'], df['water面积'], label='表面融水面积', )
# ax1.set_ylabel('面积', fontsize=20)
# ax1.legend(loc=1,fontsize=15)
# plt.yticks(fontsize=13)
#
# ax2 = ax1.twinx()
# ax2

plt.ylim((0, 100))
plt.xlabel('年份')
plt.ylabel(r'面积($km^2$)')

plt.bar(df['年份'], df['water面积'], label='表面融水面积', width=0.45, color='#0089BA')
plt.plot(df['年份'], df['water面积'], label='表面融水面积', color='darkgrey', ms=10, mfc='black', lw=3, marker='o')
for x, y in zip(df['年份'], df['water面积']):
    plt.text(x, y + 2, y, ha='center', va='bottom', fontsize=15)
plt.legend(loc=1, fontsize=15)
plt.yticks(fontsize=13)

plt.savefig('D:\\data\\s-2\\Amery冰架2017-2022年1月份表面融水面积变化图.svg', format='svg')
