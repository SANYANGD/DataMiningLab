import pandas as pd
import numpy as np
from sklearn import preprocessing



#pandas.read_csv似乎不能直接获取到表头信息和列数
headerInfo=["age","anaemia","creatinine","diabetes"
    ,"ejection_fraction","high_blood_pressure","high_blood_pressure","platelets",
    "serum_creatinine","serum_sodium","sex","smoking","time","DEATH_EVENT"]
cuttedData=[]  #被分成十分的数据
trainData=[]
testData=[]
yangbenshu=10
def initData():
# data是一个数组,根据表头读取具体的列，之后就是一个数组的形式了 
    data=pd.read_csv("E:\\大学课程资源\软件\\3.1数据挖掘"
                     "\\实验三：基于SVM的医学数据分类\\heart_failure_clinical_records_dataset.csv")


    #将数据随机化
    randomedData = pd.DataFrame.sample(data,frac=1)
    zscore=preprocessing.MinMaxScaler()
    standradedRandomData =zscore.fit_transform(randomedData)


    h=0
    for i in range(0,yangbenshu):
        m= int(len(randomedData)/yangbenshu)
        if i== yangbenshu-1:
            object=standradedRandomData[h:]
        else:
            object=standradedRandomData[h:h+m]
        cuttedData.append(object)
        count=len(cuttedData[i])
        for j in range(0,count):
            if(cuttedData[i][j][12] == 0):
                cuttedData[i][j][12]=-1
        h=h+m

#将cut之后的十份数据划分成两个类别
def initUsableData(input):
    trainData.clear()
    testData.clear()
    #需要注意的是range(min,max)得到的序列当中是不包含max的
    for i in range(0,yangbenshu):
        if(i ==input):
            for j in range(0,len(cuttedData[i])):
                testData.append(cuttedData[i][j])
        else:
            for j in range(0, len(cuttedData[i])):
                trainData.append(cuttedData[i][j])










