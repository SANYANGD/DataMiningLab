import fileRead
import svm
fileRead.initData()
fileRead.initUsableData(0)

print(len(fileRead.trainData))
print(fileRead.trainData[269][0:svm.weidu-1])