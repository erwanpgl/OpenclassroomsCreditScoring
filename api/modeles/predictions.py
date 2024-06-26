import numpy as np
import pandas as pd
import pickle
#from modeles.preprocesser import featurePreprocessing
import time
import os
import shap

print (os.getenv('PA_USERNAME'))
print (os.getenv('PYTHONANYWHERE_DOMAIN'))
print (os.getenv('CSV_NAMES'))

if os.getenv('CSV_NAMES') == "reduced_for_tests": #tests launched by github
    server_path_modeles = "api/modeles/"
    server_path_files = "api/fichiers_csv/"
    files_name_end = "_4tests"
elif os.getenv('PYTHONANYWHERE_DOMAIN') == "pythonanywhere.com": #case deployed on pythonanywhere =
    server_path_modeles = "mysite/api/modeles/"
    server_path_files = "mysite/api/fichiers_csv/"
    files_name_end = "_production"
else: #local
    server_path_modeles = "api/modeles/"
    server_path_files = "C:/Users/erwan/openclassroomsRessources/projet7/Projet+Mise+en+prod+-+home-credit-default-risk/"
    files_name_end = ""

print(os.listdir())
print(server_path_modeles)

path_lightgbm = server_path_modeles + 'model_lightgbm.pkl'

path_model = path_lightgbm

print(os.listdir(os.curdir))
print(path_model)
os.environ['OMP_NUM_THREADS'] = '1'
# Load pipeline and model using the binary files
model = pickle.load(open(path_model, 'rb'))
#pipeline = pickle.load(open('pipeline.pkl', 'rb'))

#load data and calls preprocessing
#reduce_size = False #used for deploying on api site where there is a size limit
num_rows = None
path_application = server_path_files + "application_train_feature_engineered.csv"
df = pd.read_csv(path_application, nrows = num_rows)

print("application_train shape: " + str(df.shape) )

# Initialize the SHAP explainer
X_complet = df.drop(['TARGET','SK_ID_CURR'], axis=1)
explainer = shap.TreeExplainer(model)




print('fin du preprocessing et de la preparation shap')
#print("{} - faite en {:.0f}s".format("anlyse shap", time.time() - time_debut))

def predict(id):
    '''Fonction de prédiction utilisée par l\'API :
    a partir de l'identifiant 
    renvoie la prédiction à partir du modèle'''
    
    X = df[df['SK_ID_CURR'] == id]
    X = X.drop(['TARGET','SK_ID_CURR'], axis=1)
    cols = X.columns
    X= X.to_numpy()
    
    #prediction = model.predict(X) 
    time_debut = time.time()
    print("avant prédiction")
    proba = model.predict_proba(X)
    print('{}, temps écoulé: {} s'.format('prédictions', time.time() - time_debut))

    top_10_features = get_10_most_features(X, cols)

    if proba[0][0] > 0.5:
        return 0, proba, top_10_features
    else:
        return 1, proba, top_10_features

    #return prediction, proba
    
def get_10_most_features(X_sample, cols):
    # Calculates and filter the SHAP values  for 10 most important features
    time_debut = time.time()
    # Calculate SHAP values for the single instance
    shap_values = explainer(X_sample)
    # Extract SHAP values for the single instance
    shap_values_single = shap_values.values[0]  # Assuming X_sample has shape (1, n_features)
    # Create a DataFrame with feature names and their corresponding SHAP values
    shap_importance_single = pd.DataFrame(list(zip(cols, shap_values_single)),
                                      columns=['Feature', 'SHAP Value'])
    # Sort features by the absolute SHAP values
    shap_importance_single['Abs SHAP Value'] = np.abs(shap_importance_single['SHAP Value'])
    shap_importance_single_sorted = shap_importance_single.sort_values(by='Abs SHAP Value', ascending=False)
    # Select the top 10 features
    top_10_features = shap_importance_single_sorted.head(10)
    #get feature values for all clients:
    top_10_features['Mean_all_customers'] = [X_complet[feature].mean() for feature in top_10_features['Feature'].values.tolist()]
    #clients en règle
    top_10_features['Mean_all_real_solvable_customers'] = [df[df['TARGET'] == 0][feature].mean() for feature in top_10_features['Feature'].values.tolist()]
    #clients en règle
    top_10_features['Mean_all_real_defaut_customers'] = [df[df['TARGET'] == 1][feature].mean() for feature in top_10_features['Feature'].values.tolist()]
    print('{}, temps écoulé: {} s'.format('analyse shap', time.time() - time_debut)) 
    return top_10_features
    