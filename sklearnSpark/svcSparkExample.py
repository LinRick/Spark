from sklearn import svm, grid_search, datasets
from spark_sklearn import GridSearchCV
from pyspark import SparkConf, SparkContext

iris = datasets.load_iris()

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

sc = SparkContext()
#spark = SparkSession.builder.getOrCreate()


svr = svm.SVC()

clf = GridSearchCV(sc, svr, parameters)

clf.fit(iris.data, iris.target)
