import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
        函数说明：数据导入函数
        参数：fileName: 数据存放的路径
        返回：dataMat:  数据特征集
             labelMat: 数据标签集
    """
    with open(fileName) as f:
        numFeat = len(f.readline().split('\t')) - 1
        dataMat = []
        labelMat = []
        for line in f.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curLine[i]))
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("Thin matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k = 1.0):
    """
        函数说明：利用局部加权回归求解回归系数w,并且进行预测
        参数：
            testPoint : 测试样本点
            xArr :      x数据集
            yArr :      y数据集
            k :         高斯核的k,默认k=1.0
        返回：
            ws * testPoint : 测试样板点的测试结果
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))  # 创建对角线为1的矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("Thin matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k = 1.0):
    """
    函数说明：局部加权回归测试
    参数：
        testArr: 测试集
        xArr:    x数据集
        yArr:    y数据集
        k:       高斯核中的k,默认为1.0
    返回：
        yHat:    测试结果集
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat



def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("Thin matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTeatPts = 30
    wMat = np.zeros((numTeatPts, np.shape(xMat)[1]))
    for i in range(numTeatPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


if __name__ == '__main__':
    # # 线性回归主函数
    # ws = standRegres(xArr, yArr)
    # xMat = np.mat(xArr)
    # yMat = np.mat(yArr)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    # xCopy = xMat.copy()
    # xCopy.sort(0)
    # yHat = xCopy * ws
    # ax.plot(xCopy[:, 1], yHat)
    # plt.show()
    # co = np.corrcoef(yHat.T, yMat)
    # 局部加权线性回归主函数
    # xArr, yArr = loadDataSet('ex0.txt')
    # yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    # xMat = np.mat(xArr)
    # yMat = np.mat(yArr)
    # sortInd = xMat[:, 1].argsort(0)  # .argsort(0)返回升序排列索引
    # xSort = xMat[sortInd][:, 0, :]  # xMat[sortInd]有3个维度，形式为[[[]],[[]]]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0], s=2, c='red')
    # ax.plot(xSort[:, 1], yHat[sortInd])
    # plt.show()
    # 岭回归主函数
    abX, abY = loadDataSet('abalone.txt')
    a = np.array(abX)
    b = np.array(abY).T
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()
