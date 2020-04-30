import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# splitting 2 features together randomly into train and test sets
def divide(feature1, feature2, labels1, labels2):
	n = len(feature1[0])
	feature = np.hstack((feature1, feature2))
	label = labels1
	
	X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.2)

	X_train1 = X_train[:, :n]
	X_train2 = X_train[:, n:]
	X_test1 = X_test[:, :n]
	X_test2 = X_test[:, n:]
	y_train1 = y_train2 = y_train
	y_test1 = y_test2 = y_test

	return X_train1, X_train2, X_test1, X_test2, y_train1, y_train2, y_test1, y_test2


# splitting 3 features together randomly into train and test sets
def divide_triple(feature1, feature2, feature3, labels1, labels2, labels3):
	n1 = len(feature1[0])
	n2 = len(feature2[0]) + n1
	feature = np.hstack((feature1, feature2, feature3))
	label = labels1
	
	X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.2)

	X_train1 = X_train[:, :n1]
	X_train2 = X_train[:, n1:n2]
	X_train3 = X_train[:, n2:]
	X_test1 = X_test[:, :n1]
	X_test2 = X_test[:, n1:n2]
	X_test3 = X_test[:, n2:]
	y_train1 = y_train2 = y_train3 = y_train
	y_test1 = y_test2 = y_test3 = y_test

	return X_train1, X_train2, X_train3, X_test1, X_test2, X_test3, y_train1, y_train2, y_train3, y_test1, y_test2, y_test3


# success prediction function
def SPF(X_val, y_val, clf, value):
	svm_pred = clf.predict_proba(X_val)
	y = []
	for i in range(len(svm_pred)):
		y.append(np.argmax(svm_pred[i]).astype(np.float32) + 1)	

	for i in range(len(y)):
		if(y[i]==y_val[i]):
			y[i]=1
		else:
			y[i]=0

	if(value==1):
		reg = SVR(kernel='rbf', C = 0.1, gamma=1e-05).fit(X_val, y)
	elif(value==2):
		reg = SVR(kernel='rbf', C = 1, gamma=0.0001).fit(X_val, y)

	return reg


def one_hot(labels, output_dim):
	labels = list(labels)
	
	vector = np.zeros((len(labels), output_dim), dtype=int)
	for i in range(len(labels)):
		vector[i][int(labels[i])-1]=1

	return vector


# function for grid searching SVM parameters for individual features
def SVM_grid(dataset):
	seed = 7
	np.random.seed(seed)
	
	labels= dataset[:,-1]

	p = np.unique(labels)
	output_dim = len(p)
	feature = np.delete(dataset, -1, 1)

	no_of_samples, input_dimension = feature.shape
	print(no_of_samples, input_dimension)
	
	X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size = 0.2)

	# searching for best SVM parameters over the followinf range
	parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
	grid_search = GridSearchCV(SVC(), parameters)
	grid_search.fit(feature, labels)
	print(grid_search.best_params_)
	print("Best score: ", grid_search.best_score_*100)


# training a support vector machine for individual features
def SVM(dataset):
	seed = 7
	np.random.seed(seed)
	
	labels= dataset[:,-1]

	p = np.unique(labels)
	output_dim = len(p)

	feature = np.delete(dataset, -1, 1)

	no_of_samples, input_dimension = feature.shape
	print(no_of_samples, input_dimension)
	
	X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size = 0.2)
	
	# training a SVM classifier
	# grid search the parameters ('kernel', 'C', 'gamma')
	#svm_model = SVC(kernel = 'linear', C = 0.00001, gamma = 1e-05).fit(X_train, y_train)#KTH-motion
	svm_model = SVC(kernel = 'linear', C = 0.00001, gamma = 1e-05).fit(X_train, y_train)#KTH-motion+HOG
	#svm_model = SVC(kernel = 'rbf', C = 100000, gamma = 0.001).fit(X_train, y_train)#KTH-context

	svm_predictions = svm_model.predict(X_test)

	# model accuracy for X_test
	train_accuracy = svm_model.score(X_train, y_train)
	test_accuracy = svm_model.score(X_test, y_test)
 
	# creating a confusion matrix
	cm = confusion_matrix(y_test, svm_predictions)
	cm = cm.astype(np.float32)

	# normalising the confusion matrix
	for i in range(len(cm)):
		s = sum(cm[i])
		for j in range(len(cm[i])):
			cm[i][j]/=s;
	
	# print confusion matrix, training and testing accuracy				
	print(cm)
	print("Training Accuracy: ", train_accuracy*100, end='%\n')
	print("Testing Accuracy: ", test_accuracy*100, end='%\n')


# ensemble of classifiers based on maximum score/classifier weights
def Ensemble(dataset1, dataset2):
	seed = 7
	np.random.seed(seed)
	
	labels1 = dataset1[:,-1]
	labels2 = dataset2[:,-1]
	p = np.unique(labels1)
	output_dim = len(p)

	feature1 = np.delete(dataset1, -1, 1)
	feature2 = np.delete(dataset2, -1, 1)

	print(feature1.shape)
	print(feature2.shape)

	no_of_samples, input_dimension = feature1.shape
	print(no_of_samples, input_dimension)
	
	X_train1, X_train2, X_test1, X_test2, y_train1, y_train2, y_test1, y_test2 = divide(feature1, feature2, labels1, labels2)

	# training a SVM classifier
	svm_model1 = SVC(kernel = 'linear', C = 0.00001, gamma = 1e-05, probability = True).fit(X_train1, y_train1)#KTH-motion
	svm_predictions1 = svm_model1.predict_log_proba(X_test1)

	#svm_model = SVC(kernel = 'linear', C = 1).fit(X_train1, y_train1)
	#svm_pred = svm_model.predict(X_test1)	

	svm_model2 = SVC(kernel = 'rbf', C = 100000, gamma = 0.001, probability = True).fit(X_train2, y_train2)#KTH-context
	svm_predictions2 = svm_model2.predict_log_proba(X_test2)	

	svm_predictions = []
	a = []
	for i in range(len(svm_predictions1)):
		#print(np.argmax(svm_predictions[i]))
		temp = []
		for j in range(len(svm_predictions1[i])):
			# in order to take maximum of the 2 scores (default)
			temp.append(max(svm_predictions1[i][j], svm_predictions2[i][j]))

			# in order to take a weighted classifier output (weights are hyperparameters)
			#temp.append(svm_predictions1[i][j]*0.7 + svm_predictions2[i][j]*0.3)
		temp = np.array(temp)
		a.append(temp)
		x = np.argmax(temp).astype(np.float32) + 1
		svm_predictions.append(x)		

	# creating a confusion matrix
	cm = confusion_matrix(y_test1, svm_predictions)
	cm = cm.astype(np.float32)

	#normalising the confusion matrix
	for i in range(len(cm)):
		s = sum(cm[i])
		for j in range(len(cm[i])):
			cm[i][j]/=s;

	# print confusion matrix, training and testing accuracy				
	print(cm)
	test_accuracy = (cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4]+cm[5][5])/6
	print("Testing Accuracy: ", test_accuracy*100, end='%\n')
