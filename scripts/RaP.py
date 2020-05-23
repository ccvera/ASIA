#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, sys, glob, os.path, warnings, datetime, argparse, pickle

import numpy as np
import pandas as pd
from scipy import stats

from sklearn import model_selection
from sklearn import utils
from sklearn.preprocessing import label_binarize, LabelEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, LassoCV, Lasso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM, MLPRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.externals import joblib

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline

def train(classifier,X_train, X_test, Y_train, Y_test, X_validation, Y_validation):

	logging.debug("Comienza el entrenamiento")
	models = []
	
	models.append(('KNN',KNeighborsClassifier(3)))
	models.append(('LinearSVM',SVC(kernel="linear", C=0.025)))
	models.append(('RBF',SVC(gamma=2, C=1)))
	models.append(('DecisionTree',DecisionTreeClassifier(max_depth=5)))
	models.append(('RandomForest',RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
	models.append(('NeuralNetwork',MLPClassifier(alpha=1, max_iter=1000)))
	models.append(('AdaBoost',AdaBoostClassifier()))
	models.append(('NaiveBayes',GaussianNB()))
	models.append(('QDA',QuadraticDiscriminantAnalysis()))
	
	models.append(('LoR', LogisticRegression()))
	#models.append(('LiR', LinearRegression()))
	models.append(('SGD', SGDClassifier()))
	#models.append(('Lasso', Lasso()))
	#models.append(('LassoCV', LassoCV()))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('MLP', MLPClassifier(alpha=1)))
	#models.append(('MPLR', MLPRegressor(alpha=1)))
	#models.append(('BRBM', BernoulliRBM()))
	models.append(('SVM', LinearSVC(random_state=0)))
	models.append(('OVR', OneVsRestClassifier(LinearSVC(random_state=0))))
	models.append(('MBKM', MiniBatchKMeans()))
	models.append(('GP', GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)))

	for name, clf in models:
		if name in classifier:
			#gridSearch = GridSearch(clf, X_train, Y_train, X_test, Y_test) ##################################3
			#logging.debug("Score for %s model: %f \n", name, gridSearch) ##################################3

			start_time = datetime.datetime.now()
			logging.debug("Training %s classifier..." % name)
			clf.fit(X_train, Y_train)
			end_time = datetime.datetime.now()

			model_path = "models/"
        		model_name = name + ".pkl"
	        	pickle.dump(clf, open(model_path + model_name, 'wb'))
			#logging.debug("Model %s.pkl saved" % name)
			print('Duration: {}'.format(end_time - start_time))
			score = clf.score(X_test, Y_test)
			logging.debug("Score TRAIN for %s model: %f \n", name, score)
			predict_model(name, X_validation, Y_validation)
		else:
			logging.debug("Clasificador %s no seleccionado", name)

def predict(X_test, Y_test, X_validation, Y_validation):
        logging.info("Calculating predictions...")
        models = os.listdir("models/")

        for m in models:
                #logging.debug("Loading model %s", m)
                loaded_model = pickle.load(open("models/" + m, 'rb'))

                predictions = loaded_model.predict(X_validation)
                score = accuracy_score(Y_validation, predictions)
                logging.debug("Score VALIDATION for %s model: %f", m, score)

def predict_model(m, X_validation, Y_validation):
	loaded_model = pickle.load(open("models/" + m + ".pkl", 'rb'))

	predictions = loaded_model.predict(X_validation)
	score = accuracy_score(Y_validation, predictions)
        logging.debug("Score VALIDATION for %s model: %f", m, score)

def readInputFile(pattern):
	df = pd.read_csv(pattern)
    	return df

def splitDataset(df, validation=False):
    	logging.info("Spliting dataset...")

    	# Split-out validation dataset
	df.head()
    	X = df.drop(['DATE','TIMESTAMP','PRECIPITACION','LLUVIA','RANGO','PRECIPITACION_WRF','LLUVIA_WRF','RANGO_WRF'], axis=1)	# Cojo todas las features salvo la fecha y los datos de observacion
    	Y = df.LLUVIA			         					# Cojo la variable binaria 'LLUVIA'
	#Y = df.RANGO
	#Y = df.PRECIPITACION
	
    	if validation is True:
        	return X, Y
    	else:
        	X_train, X_test, Y_train, Y_test = get_X_Y(X, Y)
        	return X_train, X_test, Y_train, Y_test

def get_X_Y(X, Y):
	seed = 42
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.2, shuffle=True, random_state=seed)
        return X_train, X_test, Y_train, Y_test

def reductFeatures(df):
	logging.info("Kbest reduction started")

        # Split-out validation dataset
        df.head()
        X = df.drop(['DATE','TIMESTAMP','PRECIPITACION','LLUVIA','RANGO'], axis=1)      # Cojo todas las features salvo la fecha y los datos de observacion
        #Y = df.LLUVIA                                                                   # Cojo la variable binaria 'LLUVIA'
	Y = df.RANGO

	new_df = pd.DataFrame(SelectKBest(f_regression, k=2).fit_transform(X, Y))
	new_df['RANGO'] = Y
	X = new_df.drop(['RANGO'], axis=1)
	Y = new_df.RANGO

	logging.info("Kbest reduction finished")
	#new_df['LLUVIA'] = Y
	#new_df['PRECIPITACION'] = Y

        X_train, X_test, Y_train, Y_test = get_X_Y(X, Y)
        return X_train, X_test, Y_train, Y_test
	
def GridSearch(clf, X, y, Xt, yt):
	params = {"criterion" : ['gini', 'entropy'],
		"splitter" : ['best', 'random'],
		"max_depth" : [1 , 2, 3, 4, 5, 6],  #comprobar rangos no valores concretos
	        "min_samples_leaf" : [1 , 2, 3, 4, 5, 6],
        	"min_weight_fraction_leaf" : [0 , 0.5],
        	"max_features" : ["auto", "sqrt", "log2", 1 , 2, 3, 4, 5, 6, 1.0 , 2.0, 3.0, 4.0, 5.0, 6.0], #rango floats, rango ints
	        "random_state" : [1 , 2, 3, 4, 5, 6],
        	#"max_leaf_nodes": ['None' , 2, 3, 4, 5, 6],
        	"min_impurity_decrease" : [1.0 , 2.0, 3.0, 4.0, 5.0, 6.0]
	}

	clf = GridSearchCV(clf, param_grid=params, scoring = 'roc_auc')
	clf.fit(X, y)
	return clf.score(Xt, yt)

def custom_formatwarning(msg, *a):
	# ignore everything except the message
    	prefix = "WARNING! "
    	return prefix + str(msg) + '\n'

def main():
	warnings.formatwarning = custom_formatwarning

	logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="[%(asctime)s] %(levelname)5s: %(message)s") # timestamp + debug level + messsage

	# Análisis de argumentos
    	parser = argparse.ArgumentParser()
	parser.add_argument('pattern', help='files search pattern', type=str)
    	parser.add_argument('-e', '--evaluate', help='Evaluate all available classiffiers', default=False, dest='evaluate', action='store_true')
	parser.add_argument('-t', '--train', help='Train a MODEL using all classiffiers', default=None)
    	parser.add_argument('-m', '--model', help='Predict using MODEL classiffier', default=None)
    	parser.add_argument('-v', '--validation', help='Validates using files obtained by VALIDATION pattern', default=None)
    	parser.add_argument('-l', '--load', help='load previous saved model', default=None)
	parser.add_argument('-D', '--description', help='Description', default=None)
    	args = parser.parse_args()

    	logging.info("Reading input files...")

	df = readInputFile(args.pattern)
	X_train, X_test, Y_train, Y_test = splitDataset(df)

	if args.validation is not None:
		dfv = readInputFile(args.validation)
		X_validation, Y_validation = splitDataset(dfv, validation=True
)
	#X_train, X_test, Y_train, Y_test = reductFeatures(df)
	'''
	if args.train is not None:
		logging.info("Starting training proccess...")
		train(X_train,X_test,Y_train,Y_test)
	
        if args.validation is not None:
                logging.info("Reading validation files...")

                dfv = readInputFile(args.validation)
                X_validation, Y_validation = splitDataset(dfv, validation=True)
                predict(X_test, Y_test, X_validation, Y_validation)
	'''
	if args.description is not None:
                logging.info("#############################################")
		logging.info("#############################################")
		logging.info(args.description)
                logging.info("#############################################")
                logging.info("#############################################")


	if args.model is not None:	# Train + Test
		models = args.model.split(',')
		train(models,X_train,X_test,Y_train,Y_test,X_validation,Y_validation)

	'''
        if args.validation is not None:
                logging.info("Reading validation files...")

                dfv = readInputFile(args.validation)
                X_validation, Y_validation = splitDataset(dfv, validation=True)
                predict(X_test, Y_test, X_validation, Y_validation)
	'''

if __name__ == "__main__":
	main()
