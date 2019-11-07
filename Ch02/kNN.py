
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:47:17 2019

@author: Jason
"""

import numpy as np
import operator


def create_data_set():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify(inx, data_set, labels, k):
    # 确定已知数据行数
    data_set_size = data_set.shape[0]
    
    # 计算当前点与已知点距离
    diff_mat = np.tile(inx, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    
    # 将距离进行升序排序，返回索引值
    sorted_dist_indicies = distances.argsort()
    
    # 对前k个最近距离数据的标签进行计数
    class_count = {}
    for i in range(k):
        vote_ilabel = labels[sorted_dist_indicies[i]]
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1
        
    # 对计数结果排序，并返回最终结果
    sorted_class_count = sorted(class_count.items(), 
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file_to_matrix(file_name):
    """将文件转换为数据矩阵和标签向量"""
    # 打开文件
    with open(file_name) as file_object:
        array_lines = file_object.readlines()  # 将文本每一行存为列表的一个元素

    # 创建一个与文件行数相同的3列空矩阵
    number_lines = len(array_lines)
    data_mat = np.zeros((number_lines, 3))

    # 解析文件数据到列表
    label_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()
        list_line = line.split('\t')
        data_mat[index, :] = list_line[0:3]
        label_vector.append(int(list_line[-1]))
        index += 1

    # 返回数据矩阵和标签向量
    return data_mat, label_vector


def auto_norm(data_set):
    """数据矩阵特征值归一化"""
    # 求每列数据的最小值、最大值和范围
    min_values = data_set.min(0)
    max_values = data_set.max(0)
    ranges = max_values - min_values

    # 将数据矩阵归一化
    norm_data_set = np.zeros(np.shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - np.tile(min_values, (m, 1))
    norm_data_set = norm_data_set / np.tile(ranges, (m, 1))

    # 返回归一化的数据矩阵，范围和最小值
    return norm_data_set, ranges, min_values


def dating_class_test():
    """对分类器进行测试"""
    ho_ratio = 0.1  # 测试集比例
    # 从文件读入数据
    dating_data_mat, dating_labels = file_to_matrix('datingTestSet2.txt')
    # 数据归一化
    norm_mat, ranges, min_values = auto_norm(dating_data_mat)
    row = int(norm_mat.shape[0])
    num_test_vectors = int(row * ho_ratio)
    error_count = 0.0

    for index in range(num_test_vectors):
        classifier_result = classify(norm_mat[index, :], norm_mat[num_test_vectors:row, :],
                                     dating_labels[num_test_vectors:row], 7)
        print("the classifier came back with: %d, the real answer is: %d",
              classifier_result, dating_labels[index])
        if classifier_result != dating_labels[index]:
            error_count += 1
    print("the total error rate is: %f", (error_count/float(num_test_vectors)))


if __name__ == '__main__':
    dating_class_test()