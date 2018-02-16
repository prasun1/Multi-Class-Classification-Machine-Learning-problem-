# Loading the data and other Libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Loading the dataset directly from UCI Machine Learning repository.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#with the shape property we can define how many instances (rows) and how many attributes (columns) the data contains
print(dataset.shape)


#Looking at the first 20 rows of your data
print(dataset.head(20))

#Describing each attribute based on count, mean, the min and max values as well as some percentiles.
print(dataset.describe())


# To know the number of instances (rows) that belong to each class.
print(dataset.groupby('class').size())


#univariate plots for each individual attribute
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histogram of each input variable to get an idea of the distribution
dataset.hist()
plt.show()

#Multivariate Plots to look at the interactions between the variables.
scatter_matrix(dataset)
plt.show()

#creation of validation model to check that whether the model holds good or not.We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#We will use 10-fold cross validation to estimate accuracy. This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
seed = 7
scoring = 'accuracy'

#Now we will evaluate 6 different algorithms to build the best model:Logistic Regression (LR), Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN)., Classification and Regression Trees (CART)., Gaussian Naive Bayes (NB)., Support Vector Machines (SVM).
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
  
#create a plot of the model evaluation results to compare the spread and the mean accuracy of each model  
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#making predictions for the accuracy of our validation set
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
