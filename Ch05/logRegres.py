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
    
    
   
def plot_best_fit(weights_in):
    # 载入数据
    weights = weights_in.getA()
    data_mat, label_arr = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_arr[i] == 1):
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    # 绘制决策边界
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-)
