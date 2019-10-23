import numpy as np


# 读取文件并载入数据
def load_data_set():
    data_mat = []
    label_arr = []
    filename = 'testSet.txt'
    with open(filename) as fr:
        for line in fr.readlines():
            line_array = line.strip().split()
            data_mat.append([1.0, float(line_array[0]), float(line_array[1])])
            label_arr.append(int(line_array[2]))
    return data_mat, label_arr


# 计算sigmiod函数值
def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


# 梯度上升法
def grad_ascent(data_mat_in, label_arr_in):
    data_mat = np.mat(data_mat_in)
    label_arr = np.mat(label_arr_in).transpose()  # 分类标签列向量
    m, n = np.shape(data_mat)
    alpha = 0.001  # 步长0.001
    max_cycles = 500  # 迭代次数500
    weights = np.ones([n, 1])  # 回归系数
    for k in range(max_cycles):
        predict = sigmoid(data_mat * weights)
        error = (label_arr - predict)
        weights = weights + alpha * data_mat.transpose() * error
    return weights


if __name__ == '__main__':
    data_matrix, label_array = load_data_set()
    wei = grad_ascent(data_matrix, label_array)
    print(wei)
