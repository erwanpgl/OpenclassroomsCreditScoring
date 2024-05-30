import numpy as np
import pandas as pd
import pickle
from modeles.preprocesser import featurePreprocessing
import json
import time
import os


path_lightgbm = os.getcwd() + '\\modeles\\model_lightgbm.pkl'

path_model = path_lightgbm

# Load pipeline and model using the binary files
model = pickle.load(open(path_model, 'rb'))
#pipeline = pickle.load(open('pipeline.pkl', 'rb'))

#load data and calls preprocessing
#reduce_size = False #used for deploying on api site where there is a size limit
num_rows = None
path_application = "C:\\Users\\erwan\\openclassroomsRessources\\projet7\\Projet+Mise+en+prod+-+home-credit-default-risk\\application_train.csv"
df = pd.read_csv(path_application, nrows = num_rows)

print("application_train shape: " + str(df.shape) )

df = featurePreprocessing(df)


print('fin du preprocessing et de la preparation shap')
#print("{} - faite en {:.0f}s".format("anlyse shap", time.time() - time_debut))

def predict(id):
    '''Fonction de prédiction utilisée par l\'API :
    a partir de l'identifiant 
    renvoie la prédiction à partir du modèle'''
    
    X = df[df['SK_ID_CURR'] == id]
    X = X.drop(['TARGET','SK_ID_CURR'], axis=1)
    
    #prediction = model.predict(X) 
    time_debut = time.time()
    proba = model.predict_proba(X)
    print('{}, temps écoulé: {} s'.format('prédictions', time.time() - time_debut))

    """ # Calculates the SHAP values - It takes some time
    time_debut = time.time()
    shap_values = explainer(X)
    print('{}, temps écoulé: {} s'.format('analyse shap', time.time() - time_debut)) """       

    if proba[0][0] > 0.5:
        return 0, proba, X #shap_values[0] (shap not jsonified)
    else:
        return 1, proba, X #shap_values[0]

    #return prediction, proba
    
