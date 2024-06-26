import pandas as pd
import streamlit as st
#import shap
#from streamlit_shap import st_shap
import requests
import json
import shap
import pickle
import time
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
#from PIL import Image
import base64
import plotly.graph_objects as go
#only on jupyter notebook: %matplotlib inline
#matplotlib.use('Qt5Agg') -> error

#matplotlib.use('TkAgg')
# print the JS visualization code to the notebook
#shap.initjs()

#for dev full size: path_application  = "C:\\Users\\erwan\\openclassroomsRessources\\projet7\\Projet+Mise+en+prod+-+home-credit-default-risk\\application_train.csv"
print (os.getcwd())
num_rows = None

path = ''

if "dashboard" in(os.listdir()): #used when deployed
    path = 'dashboard/'

#for prod pythinanywhere reduced size
path_application =  path + "application_train_reduced_4pythonanaywhere.csv"

@st.cache_data#mise en cache de la fonction pour exécution unique -> utiliser si utile
def chargement_liste_clients(nrows = num_rows):   
    df = pd.read_csv(path_application, nrows = num_rows)
    liste_clients = df['SK_ID_CURR'].unique()   
    return liste_clients

liste_clients = chargement_liste_clients( num_rows) 


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

#########affichage image fonds d'écran
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    .block-container {
    background-color: white
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background(path + 'images/pret_a_depenser.png')
#####################

def graphes_streamlit(df):
    '''A partir du dataframe, affichage un subplot de 10 graphes représentatif du client comparé à d'autres clients sur 10 features'''
    fig, ax = plt.subplots(4, 3, figsize=(22,22), sharex=False)
    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)    
    i = 0
    j = 0
    liste_cols = ['Client', 'Moyenne', 'En Règle', 'En défaut']
    for feature in df['Feature']:
        sns.despine(ax=None, left=True, bottom=True, trim=False)
        sns.barplot(y = df[df['Feature']==feature][['Abs SHAP Value', 'Mean_all_customers', 'Mean_all_real_solvable_customers', 'Mean_all_real_defaut_customers']].values[0],
                   x = liste_cols,
                   ax = ax[i, j])
        sns.axes_style("white")

        if len(feature) >= 18:
            chaine = feature[:18]+'\n'+feature[18:]
        else : 
            chaine = feature
        if df[df['Feature']==feature]['SHAP Value'].values[0] > 0:
            chaine += '\n(pénalise le score)'
            ax[i,j].set_facecolor('#ffe3e3') #contribue négativement
            ax[i,j].set_title(chaine, color='#990024')
        else:
            chaine += '\n(améliore le score)'
            ax[i,j].set_facecolor('#e3ffec')
            ax[i,j].set_title(chaine, color='#017320')       
        if j == 2:
            i+=1
            j=0
        else:
            j+=1
        #if i == 2:
        #    break;
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
    #if i!=2: #cas où on a pas assez de features à expliquer (ex : 445260)
        #
    #    True
    st.pyplot(fig)

    return True


def main():
    
    API_URI =  "http://127.0.0.1:5000/predict" #'http://erwanpgl.pythonanywhere.com/predict'
    
    #st.text(os.listdir()) useful for infos on server's when deployed

    st.title('Credit Solvabilité Prediction')
    
    #image = Image.open('./images/pret_a_depenser.png')
    #st.image(image)             
    id_client = st.selectbox(
    "Veuillez sélectionner le client",
    liste_clients)    

    
    predict_btn = st.button('Prédire')
    if predict_btn:
        data = [[id_client]]
        pred = None
        try:
            with st.spinner('Chargement des prédictions de solvabilité du client...'):

                pred = request_prediction(API_URI, id_client)#[0] #* 100000
                print(pred)

                API_data = json.loads(pred)
                
                prediction_num = round(API_data['proba_defaut']*100,2)
                prediction = str(round(API_data['proba_defaut']*100,2))
                message_resultat = 'Prédiction : **{0}** avec **{1}%** de risque de défaut'

                classe_predite = API_data['prediction']
                if classe_predite == 1:
                    etat = 'client à risque'
                    message_resultat = ':red[' + message_resultat.format(etat, prediction) + ']'                    
                else:
                    etat = 'client peu risqué'
                    message_resultat = ':green[' + message_resultat.format(etat, prediction) + ']'

                st.markdown(message_resultat)

                fig = go.Figure(go.Indicator(
                    mode = "number+gauge+delta",
                    gauge = {'shape': "bullet"},
                    delta = {'reference':50, 'increasing.color':'red', 'decreasing.color':'green' },
                    value = prediction_num,
                    domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},

                    
                    ))
                    #title = {'text': "Avg order size"}))
                
                st.plotly_chart(fig)
                
                #classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
                #classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
                 # (classe réelle : '+str(classe_reelle) + ')'   
                

                st.subheader("Caractéristiques ayant influencé la prédiction:")
                
                features_values = pd.read_json(API_data['top_10_features'])

                #Affichage des graphes    
                graphes_streamlit(features_values)
                
                 

        except Exception as e:
            print("An exception occurred: {}".format(e)) #.args[0]
            st.markdown(e) 
            

if __name__ == '__main__':
    main()
