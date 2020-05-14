#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging, sys, glob, os.path, warnings, datetime, argparse, pickle

import numpy as np
import pandas as pd
from scipy import stats

from sklearn import model_selection
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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.externals import joblib

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline

def train(classifier,X_train, X_test, Y_train, Y_test):

	logging.debug("ENTRO EN TRAIN")
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
			start_time = datetime.datetime.now()
			logging.debug("Training %s classifier..." % name)
			clf.fit(X_train, Y_train)

			model_path = "models/"
        		model_name = name + ".pkl"
	        	pickle.dump(clf, open(model_path + model_name, 'wb'))
			#logging.debug("Model %s.pkl saved" % name)
			end_time = datetime.datetime.now()
			print('Duration: {}'.format(end_time - start_time))
			score = clf.score(X_test, Y_test)
			logging.debug("Score for %s model: %f \n", name, score)
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

def readInputFile(pattern):
	df = pd.read_csv(pattern)
    	return df

def splitDataset(df, validation=False):
    	logging.info("Spliting dataset...")

    	# Split-out validation dataset
	df.head()
    	X = df.drop(['DATE','TIMESTAMP','PRECIPITACION','LLUVIA'], axis=1)	# Cojo todas las features salvo la fecha y los datos de observacion
    	Y = df.LLUVIA         					# Cojo la variable binaria 'LLUVIA'

    	if validation is True:
        	return X, Y
    	else:
        	seed = 42
        	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=.2, shuffle=True, random_state=seed)
        	return X_train, X_test, Y_train, Y_test

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
    	args = parser.parse_args()

    	logging.info("Reading input files...")

	df = readInputFile(args.pattern)
	X_train, X_test, Y_train, Y_test = splitDataset(df)
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
	if args.model is not None:	# Train + Test
		models = args.model.split(',')
		train(models,X_train,X_test,Y_train,Y_test)			

        if args.validation is not None:
                logging.info("Reading validation files...")

                dfv = readInputFile(args.validation)
                X_validation, Y_validation = splitDataset(dfv, validation=True)
                predict(X_test, Y_test, X_validation, Y_validation)

if __name__ == "__main__":
	main()
