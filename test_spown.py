import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 加载图片
img = mpimg.imread('car_image.png')  # 确保这里的路径指向你的图片文件

fig, ax = plt.subplots()

# 使用extent参数设置图片位置和大小
# 使用alpha参数设置透明度
# ax.imshow(img, extent=[20, 120, 20, 80], alpha=0.5)
#! use cernten point to set the image, and set the size of the image
ax.imshow(img, extent=[0, 100, 0, 100], alpha=0.5)


plt.show()
