import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import csv

#measure accuracy of model
def evaluate(model, test_features, test_labels):
	Y_pred=model.predict(test_features)
	acc = metrics.accuracy_score(test_labels, Y_pred)
	#print("Accuracy:", acc)
	return acc

#hyper parameter general and fine tuning
def hyperParam(X, Y):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) #split trainOnMeFile in training and testing data

	clf=RandomForestClassifier()

	base_model = RandomForestClassifier(n_estimators = 100)
	base_model.fit(X_train,Y_train)
	base_accuracy = evaluate(base_model, X_test, Y_test)
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
	# Number of features to consider at every split
	max_features = ['auto', 'sqrt']
	# Maximum number of levels in tree
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	# Minimum number of samples required to split a node
	min_samples_split = [2, 5, 10]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]

	random_grid = {'n_estimators': n_estimators,
				   'max_features': max_features,
				   'max_depth': max_depth,
				   'min_samples_split': min_samples_split,
				   'min_samples_leaf': min_samples_leaf,
				   'bootstrap': bootstrap}


	clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 10, cv = 4, verbose=2, random_state=42, n_jobs = -1)

	#Train the model
	clf_random.fit(X_train,Y_train)

	#choose best parameters
	best_random = clf_random.best_estimator_
	print(best_random)

	#evaluate accuracy
	random_accuracy = evaluate(best_random, X_test, Y_test)
	print(random_accuracy)

	#fine tune model by constrincting parameters based on best parameters found 
	param_grid = {
	    'bootstrap': [False],
	    'max_depth': [50, 60, 70, 80, 90],
	    'max_features': [2, 3, 4],
	    'min_samples_leaf': [2, 3, 4],
	    'min_samples_split': [2, 3, 4],
	    'n_estimators': [400, 500, 600, 700, 800]
	}

	rf = RandomForestClassifier()

	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
	                          cv = 3, n_jobs = -1, verbose = 2)

	grid_search.fit(X_train,Y_train)

	best_grid = grid_search.best_estimator_
	grid_accuracy = evaluate(best_grid, X_test, Y_test)

	print(clf_random.best_params_)
	print(grid_search.best_params_)

	#measure performance accuracy of base model and parameter models
	print(base_accuracy)
	print(random_accuracy)
	print(grid_accuracy)
	print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
	print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
	print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - random_accuracy) / random_accuracy))

#compare mean accuracy over 100 itteration run between base model and fine tuned best model parameters
def hyperCompare(X, Y):
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3) #split trainOnMeFile in training and testing data
	i = 0
	bA = 0

	while i < 100:
		base_model = RandomForestClassifier(n_estimators = 100)
		base_model.fit(X_train,Y_train)
		base_accuracy = evaluate(base_model, X_test, Y_test)
		bA += base_accuracy
		i+=1

	j = 0
	hA = 0
	while j < 100:
		hyper_model = RandomForestClassifier(bootstrap=False, max_depth=50, max_features=4, min_samples_leaf=2, min_samples_split=4, n_estimators=800)
		hyper_model.fit(X_train,Y_train)
		hyper_accuracy = evaluate(hyper_model, X_test, Y_test)
		hA += hyper_accuracy
		j+=1

	bA = bA/100
	hA = hA/100
	print(bA, hA)

def drop_numerical_outliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=['float64']) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh, result_type='reduce') \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)

def cleanTrain(df):
	df = df.drop(columns=df.columns[0], axis=1,) # drop id column from trainOnMe
	df = df.drop('x11', axis=1) # drop x11 as all values are true
	df = df.drop('x12', axis=1) # drop x12 as all values are false 
	df = df.replace({'?':np.nan}).dropna() #drop rows that contain bad data in one or more columns
	df['x1'] = df['x1'].astype(float)
	df["x13"].replace('0.37.46222', '37.46222', inplace=True)
	df['x13'] = df['x13'].astype(float)
	drop_numerical_outliers(df)
	df["x6"] = df["x6"].str.lower() #change to lowercase
	df = pd.get_dummies(df, columns=['x6']) # one-hot encode categorical data
	df = df.drop('x6_syster och brö', axis=1)

	return df

def cleanEval(dfeval):
	dfeval = dfeval.drop(columns=dfeval.columns[0], axis=1,) # drop id column from trainOnMe
	dfeval = dfeval.drop('x11', axis=1) # drop x11 as all values are true
	dfeval = dfeval.drop('x12', axis=1) # drop x12 as all values are false 
	dfeval["x6"] = dfeval["x6"].str.lower() #change to lowercase
	dfeval = pd.get_dummies(dfeval, columns=['x6']) # one-hot encode categorical data
	return dfeval

f= open("labels.txt","w+")
f= open("checker.csv","w+")

df = cleanTrain(pd.read_csv("TrainOnMe-3.csv"))
dfeval = cleanEval(pd.read_csv("EvaluateOnMe-3.csv")) 

X_train = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x10', 'x13', 'x6_7-11', 'x6_alfvensalen', 'x6_biblioteket', 'x6_entré', 'x6_lindstedvägen 24', 'x6_syster och bro', 'x6_östra station']]  # Training Features
Y_train = df['y']  # Train Labels
X_test = dfeval[['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x10', 'x13', 'x6_7-11', 'x6_alfvensalen', 'x6_biblioteket', 'x6_entré', 'x6_lindstedvägen 24', 'x6_syster och bro', 'x6_östra station']]  # Testing Features
X_train.to_csv(r'checker.csv', index = False)
#train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.3) #split trainOnMeFile in training and testing data

#hyperParam(X_train, Y_train) # try for best tuning of parameters (no change ended up being best)
#hyperCompare(X_train, Y_train)

clf=RandomForestClassifier(bootstrap=False, max_depth=50, max_features=4, min_samples_leaf=2, min_samples_split=4, n_estimators=800)


#Train the model using the training sets
clf.fit(X_train, Y_train)

#make predictions of the new data based on the trained model
Y_pred=clf.predict(X_test)

#write labels to a text file
#for items in Y_pred: 
    #f.write(items + "\n") 