import numpy as np


def loadSimpData():
    dataMat = np.mat([[1.0, 2.1],
                       [2.0, 1.1],
                       [1.3, 1.0],
                       [1.0, 1.0],
                       [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):
            # 包含'lt', 'gt'是因为，不确定'-1'是与'lt', 'gt'哪个对应
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predicteVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predicteVals == labelMat] = 0
                weightedError = D.T * errArr
                print('split: dim %d, thresh %.2f, thresh inequal: '
                      '%s, the weighted error is %.3f' %
                      (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predicteVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClassEst


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    D = np.mat(np.ones((5, 1)) / 5)
    bestStump, minError, bestClassEst = buildStump(dataMat, classLabels, D)
