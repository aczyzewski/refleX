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
        assert 'images_path' in parameters
        assert 'images_folder' in parameters
        assert 'test_csv' in parameters

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
        super(FastaiLearner, self).__init__(parameters)
        
        # --- Fast.ai params
        fastai_metrics = [f2, average_precision, hamming_score]
        fastai_transforms = [RandomLighting(0.1, 0.1), RandomDihedral()]
        
        # --- Learner
        self.model = globals()[self.parameters['architecture']]
        transformations = tfms_from_model(self.model, self.parameters['image_size'], aug_tfms=fastai_transforms, max_zoom=1.1)
        self.data = ImageClassifierData.from_csv(
            self.parameters['images_path'], self.parameters['images_folder'], self.parameters['validate_train_csv'], 
            tfms=transformations, val_idxs=list(range(200)), test_name='copy_test', suffix='.png', bs=self.parameters['bs']
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
        

import pandas as pd
import cv2 as cv
import numpy as np
import glob 
import sys
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier #, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.gaussian_process import GaussianProcessClassifier

#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate

        
class VectorLearner(CustomLearner):
    
    statistics = [
        "var", 
        "median", 
        "mean",
        "95th_percentile",
        "5th_percentile"
    ]
    scoring_types = [
        'recall_macro',
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc'
    ]
    class_names = [
        "Loop scattering",
        "Background ring",
        "Strong background",
        "Diffuse scattering",
        "Artifact",
        "Ice ring",
        "Non-uniform detector"
    ]
    classifier_names = [
        "RFC",
        "DTC",
        "KNN",
        "GaussianNB",
        "QuadraticDisciminantAnalysis"
    ]
    
    def __init__(self, parameters):
        
        # --- Super-class params
        super(CustomLearner, self).__init__(parameters)
        
        # --- Learner
        self.model = globals()[self.parameters['architecture']]
        transformations = tfms_from_model(self.model, self.parameters['image_size'], aug_tfms=fastai_transforms, max_zoom=1.1)
        self.data = ImageClassifierData.from_csv(
            self.parameters['images_path'], self.parameters['images_folder'], self.parameters['validate_train_csv'], 
            tfms=transformations, val_idxs=list(range(200)), test_name='copy_test', suffix='.png', bs=self.parameters['bs']
        )
        self.learner = ConvLearner.pretrained(self.model, self.data, ps=self.parameters['dropout'], metrics=fastai_metrics)
        
        if 'saved_model_path' in self.parameters.keys() and not self.parameters['saved_model_path'] is None:
            self.learner.load(self.parameters['saved_model_path'])
    
    def train(self):
        pass
    
    def predict(self, image):
        trn_tfms, val_tfms = tfms_from_model(self.model, self.parameters['image_size'])
        img = val_tfms(open_image(image))
        self.learner.precompute = False 
        return self.learner.predict_array(img[None])
    
    def classify(X, y, classifier_name, scoring):    
  
        #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.1, random_state=42)
        #print(f'Training set size: {len(X_train)} & Test set size: {len(X_test)}')
        ### cv - cross-validation generator - default KFold(n_splits, shuffle, random state) splits into K folds 

        clf = get_classifier(classifier_name)
        return cross_validate(clf, X, y, cv=no_cv_folds, scoring=scoring)
        #return cross_val_score(clf, X, y, cv=no_cv_folds, scoring=scoring)
    
    '''
    Constructs a DataFrame from all vector files in the given directory.
    '''
    def vector_folder_to_df(directory, limit=None):

        files = list(os.listdir(directory + '/' + list(os.listdir(directory))[0]))
        all_dfs = []

        for idx, image in enumerate(files[:limit]): 
            all_dfs.append(create_joint_vector(image, directory, all_statistics=True, sort=True, vector_length=240, remove_suffix=True, image_as_index=True))

        return pd.concat(all_dfs)

    def extract_filename(path):
        path, filename = os.path.split(path)
        return '.'.join(filename.split('.')[:-1])

    def construct_joint_csv(output_filepath):
        df1 = pd.read_csv('/home/reflex/reflex/data/results_constant_vector_length/vectors/vectors.csv')
        df1.set_index('img', inplace=True, drop=True)

        df2 = pd.read_csv('/home/reflex/reflex/reflex.csv')
        df2['Image'] = df2['Image'].apply(lambda x: extract_filename(x))
        df2.set_index('Image', inplace=True, drop=True)

        joint = pd.concat([df1, df2], axis=1, join='inner')
        joint.to_csv(output_filepath)
        return joint
    
    def get_classifier_filename(classifier_name, class_name):

        if classifier_name not in classifier_names:
            raise Exception("Invalid classifier name!")

        return dict(zip(classifier_names, [
          result_plot_dir + "RFClf-cv"+str(no_cv_folds)+"-n_est"+str(no_estimators)+"-"+class_name+".jpg",
          result_plot_dir + "DTClf-cv"+str(no_cv_folds)+class_name+".jpg",
          result_plot_dir + "KNNClf-cv"+str(no_cv_folds)+"-neighbours-"+str(no_neighbours)+"-"+class_name+".jpg",
          result_plot_dir + "GaussClf-cv"+str(no_cv_folds)+"-"+class_name+".jpg",
          result_plot_dir + "QDAClf-cv"+str(no_cv_folds)+"-"+class_name+".jpg",
          result_plot_dir + "GaussPrc-cv"+str(no_cv_folds)+"-"+class_name+".jpg",
          result_plot_dir + "SV-cv"+str(no_cv_folds)+"-"+class_name+".jpg" 
        ]))[classifier_name]


    '''
    Constructs an appropriate classifier object for the given classifier name.
    '''
    def get_classifier(classifier_name):

        if classifier_name not in classifier_names:
            raise Exception("Invalid classifier name!")

        classifier_objects = [
            RandomForestClassifier(random_state=23, n_estimators=no_estimators, n_jobs=no_jobs),
            DecisionTreeClassifier(random_state=10),
            KNeighborsClassifier(n_neighbors=no_neighbours, n_jobs=no_jobs),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            SVC(gamma=2, C=1, probability="True")
        ]

        return dict(zip(classifier_names, classifier_objects))[classifier_name]

    def initDF(class_name):

        joined = pd.read_csv(joint_csv_path)
        print("Dataset size: ", len(joined))

        y = joined.loc[:, class_name].values
        X = joined.as_matrix(columns=joined.columns[1:-7])

        return X, y  
    
    def plot(data, classifier_name, class_name, scoring_names):    
    
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,12))
        ax1.set_title(f'{class_name}-{classifier_name}-{no_cv_folds}folds')

        x = []
        for i in range(len(data)):
            for j in range(len(data[0])):
                x.append(i+1)
        y = np.array(data).flatten()

        color = np.tile(np.arange(len(data[0])), len(scoring_names))
        scatter = ax1.scatter(x, y, s=20, c=color)
        cbar = plt.colorbar(scatter, ax=[ax1, ax2])
        cbar.set_ticks(np.arange(len(data[0])))

        ax2.boxplot(data, 0, 'gD', 1)

        ax2.set_xticklabels(scoring_names, rotation=45)

        plt.show()
        f.savefig(get_classifier_filename(classifier_name, class_name))

        
    @staticmethod
    def create_joint_vector(image, directory, statistics=None, all_statistics=False, sort=False, 
                        sort_function=lambda x: x, vector_length=None, 
                        fill_value=None, image_as_index=True, predefined_file_list=None, 
                        remove_suffix=False):

        """
        Tworzy dataframe z wybranego zdjecia w postaci <nazwa_zdjecia> <wektor_zlaczonych_statystyk>

        Params:
        --------------
        image                 - nazwa pliku ze zdjeciem (z suffixem (.SSSxSSS.png))
        directory             - katalog z katalogami ze wszystkimi statystykami
        statistics            - lista statystyk branych pod uwage
        all_statistics        - jezeli True to poprzedni parametr jest ignorowany i pod uwage bierzemy
                                wszystkie statystyki we wskazanym folderze
        sort                  - jezeli True to nazwy statystyk sortowane sa wedlug podanej funkcji
                                sortujacej (domyslnie leksykograficznie)
        sort_function         - funkcja (key) sortujaca statystyki (domyslnie leksykograficznie)
        vector_length         - parametr okreslajacy pozadana dlugosc wektora. W przypadku 
                                nadmiaru jest przycinany, w przeciwnym razie jest wypelniniany
                                kolejnym paremetrem. Niezdefiniowany (None) nie modyfikuje wektora.
        fill_value            - wartosc, ktora wypelniany bedzie wektor, jezeli będzie za krótki
                                w przypadku zdefiniowania dlugości
        image_as_index        - ustawie nazwe pliku jako index DataFrame'u
        remove_suffix         - usuwa suffix (.SSSxSSS.png) z nazwy pliku


        Returns:
        --------------
        None                  - w przpadku bledu (brak zdef. statystyk, zly katalog, zla nazwa pliku)
        Dataframe             - kiedy wszystko poszlo zgodnie z zalozeniami

        """

        statistic_names = list(os.listdir(directory)) if all_statistics else statistics 

        if statistic_names and sort:
            statistic_names.sort(key=sort_function)

        values = []

        # Laczenie wektora
        for stat_name in statistic_names:

            current_vector = cv.imread(os.path.join(directory, stat_name, image), cv.IMREAD_GRAYSCALE).flatten().tolist()

            # Dostosowywanie dlugosci wektora
            if vector_length:
                current_vector = current_vector[:vector_length]
                current_vector.extend([fill_value] * max(0, vector_length - len(current_vector)))

            values.append(current_vector)

        if remove_suffix:
            image = image[:-12]

        data = []
        column_names = ['img']
        for idx, stat_vec in enumerate(values):
            column_names += [f'{statistic_names[idx]}_{i}' for i in range(len(stat_vec))]
            data += stat_vec

        df = pd.DataFrame([[image, *data]], columns=column_names)

        if image_as_index:
            df.set_index('img', inplace=True)

        return df
