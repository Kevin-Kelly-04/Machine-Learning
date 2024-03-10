import pandas as pd 
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import math

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


	clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)

	#Train the model
	clf_random.fit(X_train,Y_train)

	#choose best parameters
	best_random = clf_random.best_estimator_
	print("best_random")

	#evaluate accuracy
	random_accuracy = evaluate(best_random, X_test, Y_test)

	#fine tune model by constrincting parameters based on best parameters found 
	param_grid = {
	    'bootstrap': [True],
	    'max_depth': [50, 60, 70, 80, 90],
	    'max_features': [2, 3, 4],
	    'min_samples_leaf': [2, 3, 4],
	    'min_samples_split': [2, 3, 4],
	    'n_estimators': [700, 800, 900, 1000]
	}

	rf = RandomForestClassifier()

	grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
	                          cv = 3, n_jobs = -1, verbose = 2)

	grid_search.fit(X_train,Y_train)

	#print(clf_random.best_params_)
	#print(grid_search.best_params_)
	best_grid = grid_search.best_estimator_
	grid_accuracy = evaluate(best_grid, X_test, Y_test)

	#measure performance accuracy of base model and parameter models
	print(base_accuracy)
	print(random_accuracy)
	print(grid_accuracy)
	print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
	print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
	print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - random_accuracy) / random_accuracy))


#preprocessing
df = pd.read_csv("TrainOnMe-3.csv") 
pd.plotting.scatter_matrix(df, alpha=0.2)
dfeval = pd.read_csv("EvaluateOnMe-3.csv") 
df = df.drop(columns=df.columns[0], axis=1,) # drop id column from trainOnMe
df = df.drop('x11', axis=1) # drop id column from trainOnMe
df = df.drop('x12', axis=1) # drop id column from trainOnMe
out = df.replace({'?':np.nan}).dropna() #drop rows that contain bad data in one or more columns
out["x6"] = out["x6"].str.lower() #change to lowercase
dfeval["x6"] = dfeval["x6"].str.lower()
out = pd.get_dummies(out, columns=['x6'])
dfeval = pd.get_dummies(dfeval, columns=['x6'])
out = out.drop('x6_syster och brö', axis=1)
out['x1'] = out['x1'].astype(float)
out["x13"].replace('0.37.46222', '37.46222', inplace=True)
out['x13'] = out['x13'].astype(float)

X_train = out[['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x10', 'x13', 'x6_7-11', 'x6_alfvensalen', 'x6_biblioteket', 'x6_entré', 'x6_lindstedvägen 24', 'x6_syster och bro', 'x6_östra station']]  # Training Features
Y_train = out['y']  # Train Labels
X_test = dfeval[['x1', 'x2', 'x3', 'x4', 'x5', 'x7', 'x8', 'x9', 'x10', 'x13', 'x6_7-11', 'x6_alfvensalen', 'x6_biblioteket', 'x6_entré', 'x6_lindstedvägen 24', 'x6_syster och bro', 'x6_östra station']]  # Testing Features

hyperParam(X_train, Y_train) # try for best tuning of parameters (no change ended up being best)
hyperCompare(X_train, Y_train)
