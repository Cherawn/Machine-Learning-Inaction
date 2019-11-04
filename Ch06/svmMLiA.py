import numpy as np

"""
函数说明:读取数据
Parameters:
    fileName - 文件名
Returns:
    dataMat - 数据矩阵
    labelMat - 数据标签
"""


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:  # 逐行读取，滤除空格等
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append(float(lineArr[0]), float(lineArr[1]))  # 添加数据
            labelMat.append(int(lineArr[2]))  # 添加标签
    return dataMat, labelMat


"""
函数说明:随机选择alpha
Parameters:
    i - alpha
    m - alpha参数个数
Returns:
    j -
"""


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


"""
函数说明:修剪alpha
Parameters:
    aj - alpha值
    H - alpha上限
    L - alpha下限
Returns:
    aj - alpha值
"""


def clipAlpua(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
函数说明:简化版SMO算法
Parameters:
    dataMatIn - 数据矩阵
    classLabels - 数据标签
    C - 松弛变量
    toler - 容错率
    maxIter - 最大迭代次数
Returns:
    无
"""


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros(m, 1))
    iter_num = 0
    while iter_num < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T) + b)
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T) + b)
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[j] != labelMat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print('L = H');continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                if eta > 0: print('eta > 0');continue
