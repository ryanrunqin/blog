from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
# sklearn 只接受数值型的值
from sklearn import tree
from sklearn.externals.six import StringIO

allElectronicsData = open(r'F:\WebProjects\blog\decision-tree\AllElectronics.csv','r')
reader = csv.reader(allElectronicsData)
headers = next(reader)

# print(headers)

# 转化成数值型的值
# {} -- 字典 （like json obj?）
featureList = []
labelList = []
for row in reader:
    labelList.append(row[len(row) - 1])
    rowDict = {}
    for i in range(1, len(row) - 1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print(featureList)
# [{'credit_rating' :'fair','age':'youth',...} {...}]

# 转换特征值
vec=DictVectorizer()
dummyX = vec.fit_transform(featureList).toarray()

# 转换class label
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

# clf 分类器
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(dummyX, dummyY)

with open('tree.dot','w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file = f)

oneRowX = dummyX[0, :]

newRowX = oneRowX

newRowX[0] = 1
newRowX[2] = 0

predictedY = clf.predict([newRowX])

