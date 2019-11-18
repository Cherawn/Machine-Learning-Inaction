import numpy as np


# 创建实验数据样本
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'pleas'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'working', 'dog', 'food', 'stupid']
                   ]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec


# 创建所有文本出现的不重复词的列表
def creatVocabList(dataSet):
    vocabset = set([])
    for document in dataSet:
        vocabset = vocabset | set(document)
    return list(vocabset)


# 将文本转化为向量
def setOfAWorks2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary!" % word)
    return returnVec


def bagOfAWorks2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClassl):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClassl)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClassl)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    listPosts, listClasse = loadDataSet()
    myVocabList = creatVocabList(listPosts)
    # print(myVocabList)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfAWorks2Vec(myVocabList, postinDoc))
    # train = np.array(trainMat)
    # list = np.array(listClasse)

    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(listClasse))

    testEntry = ['love', 'my']
    # testEntry = ['stupid', 'dog']
    thisDoc = np.array(setOfAWorks2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
