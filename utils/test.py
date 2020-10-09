import h5py
import numpy as np
import pandas as pd

imgData = []
row = 0

with open('../data/trainImages.txt', 'r', encoding='utf-8') as source:
    for line in source:
        imgData.append(line.strip())
        row += 1
# create and write
df = pd.read_hdf('../data/cityscapes_paths_train.h5', key='df').sample(frac=1).reset_index(drop=True)
t = df['path'].values
df1 = pd.read_table('../data/trainImages.txt').sample(frac=1).reset_index(drop=True)
tt = df1.values
f = h5py.File("testh.h5", 'w')  # 创建一个h5文件，文件指针是f
f.create_dataset("path", data=imgData)
# f['path'] = imgData  # 将数据写入文件的主键data下面
#f['labels'] = range(3) # 将数据写入文件的主键labels下面
f.close() # 关闭文件

# read
f = h5py.File("testh.h5", 'r') # 打开h5文件
for key in f.keys(): # 查看所有键值
    print(f[key].name)
    print(f[key].shape)
    print(f[key].value)
f.close()