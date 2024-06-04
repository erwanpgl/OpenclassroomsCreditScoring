import numpy as np
import pandas as pd
import pickle
#from api.modeles.preprocesser import featurePreprocessing
from modeles.preprocesser import featurePreprocessing
import json
import time
import os

print (os.getenv('PA_USERNAME'))
print (os.getenv('PYTHONANYWHERE_DOMAIN'))
print (os.getenv('CSV_NAMES'))

if os.getenv('CSV_NAMES') == "reduced_for_tests": #tests launched by github
    server_path_modeles = "mysite/"
    server_path_files = "mysite/fichiers_csv"
    files_name_end = "_4tests"
elif os.getenv('PYTHONANYWHERE_DOMAIN') == "pythonanywhere.com": #case deployed on pythonanywhere =
    server_path_modeles = "mysite/"
    server_path_files = "mysite/fichiers_csv"
    files_name_end = "_production"
else: #local
    server_path_modeles = "api/modeles/"
    server_path_files = "C:/Users/erwan/openclassroomsRessources/projet7/Projet+Mise+en+prod+-+home-credit-default-risk/"
    files_name_end = ""

path_lightgbm = server_path_modeles + 'model_lightgbm.pkl'

path_model = path_lightgbm

print(os.listdir(os.curdir))
print(path_model)
# Load pipeline and model using the binary files
model = pickle.load(open(path_model, 'rb'))
#pipeline = pickle.load(open('pipeline.pkl', 'rb'))

#load data and calls preprocessing
#reduce_size = False #used for deploying on api site where there is a size limit
num_rows = None
path_application = server_path_files + "application_train" +  files_name_end +  ".csv"
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
    
