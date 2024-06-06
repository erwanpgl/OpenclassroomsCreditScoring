import pandas as pd
import streamlit as st
from streamlit_shap import st_shap
import requests
import json
import shap
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
import os
#only on jupyter notebook: %matplotlib inline
#matplotlib.use('Qt5Agg') -> error

#matplotlib.use('TkAgg')
# print the JS visualization code to the notebook
#shap.initjs()

#for dev full size: path_application  = "C:\\Users\\erwan\\openclassroomsRessources\\projet7\\Projet+Mise+en+prod+-+home-credit-default-risk\\application_train.csv"

num_rows = None

#for prod pythinanywhere reduced size
path_application =  "dashboard/application_train_reduced_4pythonanaywhere.csv"

#for dev path_lightgbm = 'C:\\Users\\erwan\\projet7_modele_scoring\\credit_scoring\\api\\modeles\\model_lightgbm.pkl'
#for prod on pythonanywhere trained on reduced feataures because reduced data (kernel with all csv and feature enginneering) 
path_lightgbm = 'dashboard/model_lightgbm_reduced_4pythonanaywhere.pkl'

path_model = path_lightgbm

@st.cache_data
def chargement_shap_explainer():
    # Load pipeline and model using the binary files
    model = pickle.load(open(path_model, 'rb'))
    #shap explainer
    explainer = shap.TreeExplainer(model)
    return explainer


@st.cache_data#mise en cache de la fonction pour exécution unique -> utiliser si utile
def chargement_liste_clients(nrows = num_rows):   
    df = pd.read_csv(path_application, nrows = num_rows)
    liste_clients = df['SK_ID_CURR'].unique()   
    return liste_clients

liste_clients = chargement_liste_clients( num_rows) #[5, 2.5, 1.75, 0.15] #

explainer = chargement_shap_explainer()



def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    print(data)
    data_json = {'client_id': int(data)}

    response = requests.request(
    method='POST', headers=headers, url=model_uri, json=data_json)  

    print(response.status_code) 
    print(response.text) 
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.text


def main():
    #API_URI =  'http://127.0.0.1:5000/predict'
    API_URI = 'http://erwanpgl.pythonanywhere.com/predict'
    #CORTEX_URI = 'http://0.0.0.0:8890/'
    #RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'
    
    #st.text(os.listdir()) useful for infos on server's when deployed

    st.title('Credit Solvabilité Prediction')

    id_client = st.selectbox(
    "Veuillez sélectionner le client",
    liste_clients)

    #id_client = st.number_input('Id client', value=215354., step=1.)

    #CODE_GENDER= st.sidebar.selectbox(
    #    'Genre',
    #    ['M', 'F'])

    
    predict_btn = st.button('Prédire')
    if predict_btn:
        data = [[id_client]]
        pred = None
        try:
            with st.spinner('Chargement des prédictions de solvabilité du client...'):

                pred = request_prediction(API_URI, id_client)#[0] #* 100000
                print(pred)

                API_data = json.loads(pred)

                # Calculates the SHAP values - It takes some time
                time_debut = time.time()
                features_values = pd.read_json(API_data['features_values'])
                shap_values = explainer(features_values)
                
                print('{}, temps écoulé: {} s'.format('analyse shap', time.time() - time_debut))
        
                classe_predite = API_data['prediction']
                if classe_predite == 1:
                    etat = 'client à risque'
                else:
                    etat = 'client peu risqué'
                #proba = 1-API_data['proba'] 

                #affichage de la prédiction
                prediction = API_data['proba_defaut']
                #classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
                #classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
                message_resultat = 'Prédiction : **' + etat +  '** avec **' + str(round(prediction*100)) + '%** de risque de défaut' # (classe réelle : '+str(classe_reelle) + ')'   

                st.markdown(message_resultat) 
                
                def displayshap():
                    matplotlib.use('TkAgg')
                    shap.plots.waterfall(shap_values[0], max_display=10)
                    plt.show()

                explain_btn = st.button("Caractéristiques ayant influencé la prédiction:", on_click=displayshap)
               
                st.subheader("Caractéristiques ayant influencé la prédiction:")
                #no module plptly st.plotly_chart(shap.plots.waterfall(shap_values[0]))
                #st_shap(shap.plots.waterfall(shap_values[0]), width= 1600,  height=600)
                #shap.initjs()
                #fig = plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
                st_shap(shap.plots.waterfall(shap_values[0], max_display=10), width= 900,  height=600)
                #shap.plots.waterfall(shap_values[0], max_display=10)  # Customize max_display if desired
                #st.pyplot(fig)

        except Exception as e:
            print("An exception occurred: {}".format(e)) #.args[0]
            st.markdown(e) 
            

if __name__ == '__main__':
    main()
