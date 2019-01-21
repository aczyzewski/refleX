import pandas as pd
import numpy as np
import os 
import constants
import sys

from defined_metrics import get_predefined_metrics, get_class_specific_metrics

class CustomLearner:
    
    def __init__(self, parameters):
        """Metoda inicjująca nadklasy

        Jej głównym zadaniem jest przyjęcie parametrów w postaci 
        dictionary, które zawierać będą wszystkie parametry do działania
        pozostałych metod. Definiuje ona również podstwowe metryki, używane
        w fazie testowania modelu.
        
        Parametry obowiązkowe:
            
            [ Parametry danych ]
            - 'images_path'        ścieżkę na dysku wskazującą na glowny folder z danymi
            - 'images_folder'      ścieżkę na dysku wskazującą na podfolder z obrazami
            - 'test_csv'          ścieżkę do pliku CSV wskazującego przykłady uczące
        
        Args:
            parameters (dict): parametry modelu

        """

        if 'images_path' not in parameters:
            raise Exception("Data path not specified!")
        if 'images_folder' not in parameters:
            raise Exception("Image path not specified!")
        if 'test_csv' not in parameters:
            raise Exception("Training data csv not specified!")

        self.parameters = parameters
        self.metrics = get_predefined_metrics()
        self.class_spec_metrics = get_class_specific_metrics()
        self.name = parameters['name'] if 'name' in parameters else self.__name__
        
    # --- Metody pomocnicze ---
    def set_parameter(self, parameter, value):
        """ Pozwala nadpisać zdefiniowany wcześniej parametr modelu """
        self.parameters[parameter] = value
        
    def load_parameters(self, parameters):
        """ Pozwala załadować nowy zestaw parametrów modelu """
        self.parameters = parameters
        
    def get_name(self):
        return self.name
        
    # --- Metody związane z uczeniem/testowaniem
    def train(self):
        raise NotImplementedError
        
    def builtin_test(self, threshold=0.5):
        raise NotImplementedError
                    
    def predict(self, image):
        raise NotImplementedError
    
    def test(self, threshold=0.5, use_builtin_test=False):
        
        full_path = os.path.join(self.parameters['images_path'], os.path.join(self.parameters['images_folder']))        
        data = pd.read_csv(self.parameters['test_csv'], index_col=0)
        
        # Sortowanie alfabetyczne kolumn + po nazwach obrazow
        data = data.reindex(sorted(data.columns), axis=1)
        data = data.sort_index()
        
        
        # Uzyskanie wyników dla każdego obrazu w postaci:
        #     [[1, 0, 1...],        [1, 1, 0, ...]]
        #  < Wynik predykcji>       < Ground truth > 
        #        preds                   targs
        
        preds, targs = [], []
        
        if not use_builtin_test:
            for img_data in data.to_records():
                preds.append(list(map(lambda x: int(x), self.predict(os.path.join(os.path.join(full_path, img_data[0]) + ".png")).flatten() > threshold)))
                targs.append(list(img_data)[1:])
        else:
            # TODO
            pass
                
        metrics = {metric.__name__: metric(np.array(preds), np.array(targs)) for metric in self.metrics}
        class_spec_metrics = {metric.__name__: metric(np.array(preds), np.array(targs)) for metric in self.class_spec_metrics}
    
        return metrics, class_spec_metrics
    
    
from custom_metrics import f2, average_precision, average_recall, hamming_score
sys.path.insert(0, constants.FASTAI_PATH)
from fastai.conv_learner import * 

class FastaiLearner(CustomLearner):
   
    
    def __init__(self, parameters):
        
        # --- Super-class params
        super().__init__(parameters)
        
        # --- Fast.ai params
        fastai_metrics = [f2, average_precision, hamming_score]
        fastai_transforms = [RandomLighting(0.1, 0.1), RandomDihedral()]
        
        # --- Learner
        self.model = globals()[self.parameters['architecture']]
        transformations = tfms_from_model(self.model, self.parameters['image_size'], aug_tfms=fastai_transforms, max_zoom=1.1)
        self.data = ImageClassifierData.from_csv(
            self.parameters['images_path'], self.parameters['images_folder'], self.parameters['validate_train_csv'], 
            tfms=transformations, val_idxs=list(range(200)), test_name='original512', suffix='.png', bs=self.parameters['bs']
        )
        self.learner = ConvLearner.pretrained(self.model, self.data, ps=self.parameters['dropout'], metrics=fastai_metrics)
        
        if 'saved_model_path' in self.parameters.keys() and not self.parameters['saved_model_path'] is None:
            self.learner.load(self.parameters['saved_model_path'])
    
    def builtin_test(self, threshold=0.5):
        results = sorted(list(zip(self.data.test_ds.fnames, self.learner.predict(is_test=True) > threshold)), key=lambda x: x[0])
        print(results)
    
    def train(self):
        pass
    
    def predict(self, image):
        trn_tfms, val_tfms = tfms_from_model(self.model, self.parameters['image_size'])
        img = val_tfms(open_image(image))
        self.learner.precompute = False 
        return self.learner.predict_array(img[None])
        
        


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

import cv2 as cv

from joblib import dump, load

        
class VectorLearner(CustomLearner):
    
    # TODO these 2 dicts would be better as 1 pd.df
    available_classifiers = {
        "RF": RandomForestClassifier,
        "KNN": KNeighborsClassifier,
        "SVM": LinearSVC,
        "GaussianNB": GaussianNB
    }
    
    grid_search_params = {
        "RF": [{'n_estimators': [10, 20, 50, 100, 200]}],
        "KNN": [{'n_neighbors': [5, 7, 10, 15, 20, 30, 50]}],
        "SVM": [{'C': [1, 5, 10], 'penalty':['l2']}],
        "GaussianNB": [{'priors':[None], 'var_smoothing': [1e-5, 1e-7, 1e-9, 1e-10]}]
    }
        
    def __init__(self, parameters):
        
        assert set(self.available_classifiers.keys()) == set(self.grid_search_params.keys())
        
        if parameters['name'] not in self.available_classifiers.keys(): # TODO more checks
            raise Exception("Invalid classifier name!")
        
        # --- Super-class params
        super(VectorLearner, self).__init__(parameters)
                
        df_train = pd.read_csv(self.parameters['train_csv'], index_col=0)
        df_val = pd.read_csv(self.parameters['val_csv'], index_col=0)
        self.cv_split = [(np.arange(0, len(df_train)), np.arange(len(df_train), len(df_train) + len(df_val)))]
        self.Y = df_train.append(df_val)
        
        assert (self.Y.columns.values == df_train.columns.values).all() 
        assert (df_val.columns.values == df_train.columns.values).all()
        assert len(self.Y) == len(df_train) + len(df_val)
                    
        self.filepath = self.parameters['images_path'] + self.parameters['images_folder']
        filenames = self.Y.index.values
        self.vector_size = cv2.imread(self.filepath + filenames[0] + '.png', 0).shape[0]
        self.X = np.ndarray((len(self.Y), self.vector_size))
        for i, filename in enumerate(filenames):
            self.X[i,:] = cv2.imread(self.filepath + filename + '.png', 0).T

        self.classifiers = dict([(col, self.get_classifier(parameters['name'])) for col in self.Y.columns])
        print(self.classifiers)
        
        
    def train(self):
        
        if 'output_model_path' not in self.parameters:
            raise Exception("No model output path not specified!")
        if 'output_model_name' not in self.parameters:
            raise Exception("No model filename specified!")
        
        model_filepath = self.parameters['output_model_path']
        model_filename = self.parameters['output_model_name']
        
        for anomaly_name, clf in self.classifiers.items():
            print(anomaly_name)
            print(len(self.Y.loc[:, anomaly_name].values))
            clf.fit(self.X, self.Y.loc[:, anomaly_name].values)
            dump(clf, f'{model_filepath}{anomaly_name}_{model_filename}') 
    
    
    def predict(self, image):
        im = cv2.imread(image, 0).T
        return np.array([self.classifiers[anomaly_name].predict(im) for anomaly_name in sorted(self.Y.columns.values)])
    

    '''
    Constructs an appropriate classifier object for the given classifier name.
    '''
    def get_classifier(self, classifier_name, parameters={}):

        assert classifier_name in self.available_classifiers.keys()
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        return GridSearchCV(self.available_classifiers[classifier_name](parameters), self.grid_search_params[classifier_name], cv=self.cv_split, n_jobs=3)
    
    
    def classify(self, X, y, classifier_name, scoring): # FIXME NOT USED   
  
        #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.1, random_state=42)
        #print(f'Training set size: {len(X_train)} & Test set size: {len(X_test)}')
        ### cv - cross-validation generator - default KFold(n_splits, shuffle, random state) splits into K folds 

        clf = get_classifier(classifier_name)
        return cross_validate(clf, X, y, cv=no_cv_folds, scoring=scoring)
        #return cross_val_score(clf, X, y, cv=no_cv_folds, scoring=scoring)
