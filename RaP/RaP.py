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
from sklearn.metrics import mean_squared_error

from sklearn.externals import joblib

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression
from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.pipeline import Pipeline

def train(classifier,X_train, X_test, Y_train, Y_test, i):

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
			start_time = datetime.datetime.now()
			logging.debug("Training %s classifier..." % name)
			clf.fit(X_train, Y_train)
			end_time = datetime.datetime.now()

			#model_path = "models/binaria/todas_caracteristicas_sin_RAIN/"
			model_path = "models/test_kbest/"
        		#model_name = name + ".pkl"
			model_name = name + "_" + str(i) + ".pkl"
	        	pickle.dump(clf, open(model_path + model_name, 'wb'))
			#logging.debug("Model %s.pkl saved" % name)
			print('Duration: {}'.format(end_time - start_time))
			score = clf.score(X_test, Y_test)
			logging.debug("Score TRAIN for %s model: %f \n", name, score)
			#predict_model(name, X_validation, Y_validation)
		else:
			#logging.debug("Clasificador %s no seleccionado", name)
			print(".")

def predict(X_validation, Y_validation):
        logging.info("Calculating predictions...")
	#path = "models/rango/todas_caracteristicas/"
	path = "models/binaria/todas_caracteristicas_sin_RAIN/"
        models = os.listdir(path)

        for m in models:
                logging.debug("Loading model %s", m)
                loaded_model = pickle.load(open(path + m, 'rb'))

                predictions = loaded_model.predict(X_validation)
                score = accuracy_score(Y_validation, predictions)
                logging.debug("Score VALIDATION for %s model: %f", m, score)
	
		#print predictions
		get_mse_diario(Y_validation,predictions)
		#logging.debug("Error cuadrático medio for %s model: %f", m, mean_sqr_error(Y_validation,predictions))
		#logging.debug("Error cuadrático medio WRF - CHE: ", mean_sqr_error_CHE(Y_validation,y_train))

def get_mse_filas(Y_validation,predictions):

	acierto 	= 0
	f_negativo 	= 0
	f_positivo	= 0

	for i,t in enumerate(Y_validation):
		mse = t - predictions[i]
		if mse == 0:
			acierto += 1
		elif mse == -1:
			f_negativo += 1
		elif mse == 1:
			f_positivo += 1
	logging.debug("Numero de aciertos: %i", acierto)
	logging.debug("Numero de falsos positivos: %i", f_positivo)
        logging.debug("Numero de falsos negativos: %i", f_negativo)
	logging.debug("Total: %i", acierto+f_positivo+f_negativo)

def get_mse_diario(Y_validation,predictions):
	iters = Y_validation.shape[0] / 9594

	i = 0
	f = 5100
	for it in range(iters):
		logging.debug("Error cuadrático medio dia %s: %f", get_day(i),  mean_sqr_error(Y_validation[i:f],predictions[i:f]))
		i += 5100
		f += 5100

def get_day(i):
	df = readInputFile("/home/fcsc/ccalvo/METEO/preproces/SCIKIT-LEARN/merged_validation.csv")
	return df.DATE[i]

def readInputFile(pattern):
	df = pd.read_csv(pattern)
    	return df

def splitDataset(df, validation=False):
    	logging.info("Spliting dataset...")

    	# Split-out validation dataset
	#df.head()
	#X = df.drop(['DATE','TIMESTAMP','PRECIPITACION','LLUVIA','RANGO','PRECIPITACION_WRF','LLUVIA_WRF','RANGO_WRF'], axis=1)
    	X = df.drop(['DATE','TIMESTAMP','RAINC','RAINNC','PRECIPITACION','LLUVIA','RANGO','PRECIPITACION_WRF','LLUVIA_WRF','RANGO_WRF'], axis=1)	# Cojo todas las features salvo la fecha y los datos de observacion
	#X = df.drop(['DATE','TIMESTAMP','T_500hPa','T_700hPa','T_850hPa','QVAPOR_500','QVAPOR_700','QVAPOR_850','QCLOUD_500','QCLOUD_700','QCLOUD_850','QRAIN_500','QRAIN_700','QRAIN_850','QICE_500','QICE_700','QICE_850','QSNOW_500','QSNOW_700','QSNOW_850','QGRAUP_500','QGRAUP_700','QGRAUP_850','PRECIPITACION','LLUVIA','RANGO','PRECIPITACION_WRF','LLUVIA_WRF','RANGO_WRF'], axis=1)
    	
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

def reductFeatures(df,k,validation=False):
	logging.info("Kbest reduction started con k = %i", k)

        # Split-out validation dataset
        #df.head()
        X = df.drop(['DATE','TIMESTAMP','RAINC','RAINNC','PRECIPITACION','LLUVIA','RANGO','PRECIPITACION_WRF','LLUVIA_WRF','RANGO_WRF'], axis=1)      # Cojo todas las features salvo la fecha y los datos de observacion
        Y = df.LLUVIA                                                                   # Cojo la variable binaria 'LLUVIA'
	#Y = df.RANGO

	new_df = pd.DataFrame(SelectKBest(f_regression, k).fit_transform(X, Y))
	new_df['LLUVIA'] = Y

	selector = SelectKBest(f_regression, k)
	selector.fit(X, Y)
	# Get columns to keep and create new dataframe with those only
	cols = selector.get_support(indices=True)
	features_df_new = X.iloc[:,cols]

	pd.set_option('display.max_columns', None)	
	print(features_df_new.head(1))
	new_df = pd.DataFrame(features_df_new)
	new_df['LLUVIA'] = Y

	X = new_df.drop(['LLUVIA'], axis=1)
	Y = new_df.LLUVIA

	#print(X.head())
	
	logging.info("Kbest reduction finished")
	#	new_df['LLUVIA'] = Y
	#new_df['PRECIPITACION'] = Y

        #X_train, X_test, Y_train, Y_test = get_X_Y(X, Y)
        #return X_train, X_test, Y_train, Y_test

        if validation is True:
                return X, Y
        else:
                X_train, X_test, Y_train, Y_test = get_X_Y(X, Y)
                return X_train, X_test, Y_train, Y_test

	
def predict_model(X_validation, Y_validation, i):
        logging.info("Calculating predictions...")
        #path = "models/rango/todas_caracteristicas/"
        path = "models/test_kbest/"

	m = path + "NeuralNetwork_" + str(i) + ".pkl"
	logging.debug("Loading model %s", m)
        loaded_model = pickle.load(open(m , 'rb'))

        predictions = loaded_model.predict(X_validation)
        score = accuracy_score(Y_validation, predictions)
        logging.debug("Score VALIDATION for %s model: %f", m, score)
	
	print predictions

        logging.debug("Error cuadrático medio for %s model: %f", m, mean_sqr_error(Y_validation,predictions))

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

def mean_sqr_error(y_true, y_pred):
	return mean_squared_error(y_true, y_pred)

def custom_formatwarning(msg, *a):
	# ignore everything except the message
    	prefix = "WARNING! "
    	return prefix + str(msg) + '\n'

def featuresRanking(forest, X_train):
    importances = forest.feature_importances_
    #std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

def get_feature_NN(model, X_train, Y_train):
	feature_imp = {'feature': [], 'importance':[]}
	print X_train.shape[1]
	kbest = SelectKBest(f_regression, k=24)
	kbest = kbest.fit(X_train, Y_train)

	print(kbest.scores_)
	feature_importance = kbest.scores_
        feature_names = list(X_train.columns)
	for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
		feature_imp['feature'].append(name)
		feature_imp['importance'].append(score)

	df_fi =  pd.DataFrame(feature_imp)
	df_fi['importance'] = df_fi['importance'] / df_fi['importance'].sum()
	print df_fi
	#self.feature_importance[name_model] = df_fi

def get_feature_importance(self):
        non_tree_based_models = ['KNeighborsClassifier', 'MLPClassifier', 'SVC',
                                 'LinearDiscriminantAnalysis', 'GaussianNB',
                                 'LogisticRegression', 'KNeighborsRegressor',
                                 'MLPRegressor', 'SVR', 'LinearRegression',
                                 'BayesianRidge']

        models = self.estimators.keys()
        if self.problem_type == 'classification':
            for name_model in models:

                feature_imp = {'feature': [], 'importance':[]}
                if name_model in non_tree_based_models:
                    # estimator = self.estimators[name_model]
                    # y_pred = estimator.predict(self.X_test.values)

                    kbest = SelectKBest(score_func=chi2, k=self.X_train.shape[1])
                    kbest = kbest.fit(self.X_train, self.y_train)

                    print(kbest.scores_)
                    feature_importance = kbest.scores_
                    feature_names = list(self.X_train.columns)
                    for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
                        feature_imp['feature'].append(name)
                        feature_imp['importance'].append(score)

                    df_fi =  pd.DataFrame(feature_imp)
                    df_fi['importance'] = df_fi['importance'] / df_fi['importance'].sum()
                    self.feature_importance[name_model] = df_fi
                else:
                    # Tree based models
                    estimator = self.estimators[name_model].named_steps[name_model]
                    if not hasattr(estimator, 'feature_importances_'):
                        feature_importance = np.mean([
                            tree.feature_importances_ for tree in estimator.estimators_], axis=0)
                        feature_names = list(self.X_train.columns)
                        for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
                            feature_imp['feature'].append(name)
                            feature_imp['importance'].append(score)

                        self.feature_importance[name_model] = pd.DataFrame(feature_imp)
                    else:
                        feature_importance = estimator.feature_importances_
                        feature_names = list(self.X_train.columns)
                        for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
                            feature_imp['feature'].append(name)
                            feature_imp['importance'].append(score)

                        df_fi =  pd.DataFrame(feature_imp)
                        if name_model == 'LGBMClassifier':
                            df_fi['importance'] = df_fi['importance'] / df_fi['importance'].sum()

                        self.feature_importance[name_model] = pd.DataFrame(df_fi)


        else:
            # Here fill with regression
            self.feature_importance = None
        # print(self.metrics) 

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

	'''
	if args.validation is not None:
		dfv = readInputFile(args.validation)
		X_validation, Y_validation = splitDataset(dfv, validation=True)
	#X_train, X_test, Y_train, Y_test = reductFeatures(df)

	if args.train is not None:
		logging.info("Starting training proccess...")
		train(X_train,X_test,Y_train,Y_test)
	
        if args.validation is not None:
                logging.info("Reading validation files...")

                dfv = readInputFile(args.validation)
                X_validation, Y_validation = splitDataset(dfv, validation=True)
                predict(X_test, Y_test, X_validation, Y_validation)
	'''
	if args.load is not None:
		classiffier = joblib.load(args.load)

		#featuresRanking_LinearRegression(classiffier, X_train)
		#featuresRanking(classiffier, X_train)
		get_feature_NN(classiffier, X_train, Y_train)

	if args.description is not None:
                logging.info("#############################################")
		logging.info("#############################################")
		logging.info(args.description)
                logging.info("#############################################")
                logging.info("#############################################")


	if args.model is not None:	# Train + Test
		models = args.model.split(',')
		#train(models,X_train,X_test,Y_train,Y_test)
		for i in range(3,24):
			X_train, X_test, Y_train, Y_test = reductFeatures(df,i)
			train(models,X_train,X_test,Y_train,Y_test,i)


        if args.validation is not None:
                logging.info("Reading validation files...")

                dfv = readInputFile(args.validation)
                X_validation, Y_validation = splitDataset(dfv, validation=True)
		predict(X_validation, Y_validation)
		
		#for i in range(3,24):
		#	X_validation, Y_validation = reductFeatures(dfv,i,validation=True)
                #	predict_model(X_validation, Y_validation, i)
	
if __name__ == "__main__":
	main()
