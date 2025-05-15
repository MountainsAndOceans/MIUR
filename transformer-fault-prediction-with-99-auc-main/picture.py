import matplotlib.pyplot as plt
import numpy as np

size = 7
# 返回size个0-1的随机数
accuray = (np.random.random(size)*3+95)/100
# x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
x = np.arange(size)

# 正确显示中文和负号
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

plt.ylim(0.9, 1.0)

# 有a/b/c三种类型的数据，n设置为3
total_width, n = 0.8, 1
# 每种类型的柱状图宽度
width = total_width / n

# 重新设置x轴的坐标
x = x - (total_width - width) / 2
print(x)

# 画柱状图
plt.bar(x, accuray, width=width, color="chocolate", label="准确率")

# 功能1
x_labels = ["铁心接地", "绕组短路", "过载", "绝缘介质受损", "漏油", "套管断裂", "油劣化"]
# 用第1组...替换横坐标x的值
plt.xticks(x, x_labels)

# 功能2
for i, j in zip(x, accuray):
    plt.text(i, j + 0.001, f"{int(j*10000)/100}%", ha="center", va="bottom",fontsize=12)

plt.title("知识引导机制深度多源融合方法")

# 显示图例
plt.legend()
# 显示柱状图
plt.show()
