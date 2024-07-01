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
plt.rcParams.update({'font.size': 30})
print (os.getcwd())
num_rows = None

path = ''

if "dashboard" in(os.listdir()): #used when deployed
    path = 'dashboard/'

#for prod pythinanywhere reduced size
path_application =  path + "application_train_reduced_4pythonanaywhere.csv"

#session state inits 
if 'first_feature' not in st.session_state:
    st.session_state['first_feature'] = None  
if 'second_feature' not in st.session_state:
    st.session_state['second_feature'] = None
if 'API_data' not in st.session_state:
    st.session_state['API_data'] = ''
#if 'id_client_set' not in st.session_state:
#    st.session_state['id_client_set'] = ''
if 'added_fig' not in st.session_state:
    st.session_state['added_fig'] = None

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

#@st.cache_data#mise en cache de la fonction pour exécution unique 
#def chargement_client_top_features(client_top_features):   
#    client_top_features = client_top_features
#    return client_top_features
#
#@st.cache_data#mise en cache de la fonction pour exécution unique 
#def chargement_global_top_features(global_top_10_features):   
#    global_top_10_features = global_top_10_features
#    return global_top_10_features

def graphes_client_top_features(df):
    '''A partir du dataframe, affichage un subplot de df.count() graphes représentatif du client comparé à d'autres clients sur df.count() features'''
    nbrows = int(df.shape[0]/2) #2 graphs per row
    fig, ax = plt.subplots( nbrows, 2, figsize=(22, 11*nbrows), sharex=False)
    
    plt.subplots_adjust(hspace = 1, wspace = 0.5)    
    i = 0
    j = 0
    liste_cols = ['Client', 'Moyenne', 'En Règle', 'En défaut']
    for feature in df['Feature']:
        sns.despine(ax=None, left=True, bottom=True, trim=False)
        
        ax_current = ax[i, j] if (len(ax.shape)==2) else ax[j]            
                  
        sns.barplot(y = df[df['Feature']==feature][['Feature Client Value', 'Mean_all_customers', 'Mean_all_real_solvable_customers', 'Mean_all_real_defaut_customers']].values[0],
                    x = liste_cols,
                    ax= ax_current)
        
        sns.axes_style("white")
        
        if len(feature) >= 18:
            chaine = feature[:18]+'\n'+feature[18:]
        else : 
            chaine = feature
        if df[df['Feature']==feature]['SHAP Value'].values[0] > 0:
            chaine += '\n(pénalise la solvabilité)'
            ax_current.set_facecolor('#FD5656') #contribue négativement ffe3e3
            ax_current.set_title(chaine, color='#27010A', size=40) #990024
        else:
            chaine += '\n(améliore la solvabilité)'
            ax_current.set_facecolor('#60FD93') #e3ffec
            ax_current.set_title(chaine, color='#012E0D', size=40)        #017320
        if j == 1:
            i+=1
            j=0
        else:
            j+=1
        #if i == 2:
        #    break;
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=45)
        #plt.rc('axes', labelsize=40)    # fontsize of the x and y labels
    #en haut plt.rcParams.update({'font.size': 30})
    #if i!=2: #cas où on a pas assez de features à expliquer (ex : 445260)
        #
    #    True
    
    return fig

def graphes_callback():
    if (st.session_state.first_feature != None and st.session_state.second_feature != None
        and st.session_state.first_feature !=st.session_state.second_feature ):
        first_feature = st.session_state.first_feature
        second_feature = st.session_state.second_feature
        api_data = st.session_state['API_data']
        df_top_features = pd.read_json(api_data['client_top_features'])
        df_top_features = df_top_features[(df_top_features['Feature'] == first_feature)|(df_top_features['Feature'] == second_feature)]
        st.session_state['added_fig']  = graphes_client_top_features(df_top_features)

def graphes_global_features(df):
    feature =  [ feat.replace('_', '_\n') for feat in df['Feature']] 
    #for feat in feature:
    #    if len(feat) >= 18: 
    #        feat = feat[:18]+'\n'+feat[18:]
    shap_values = df['SHAP Importance Abs']
    shap_color = [('#FD5656' if shap_value > 0 else '#60FD93') for shap_value in df['SHAP Importance']]
    fig = plt.figure(figsize = (16, 16))
    plt.bar(feature,shap_values, color = shap_color) 
    #plt.xticks(rotation=15)        
    #sns.axes_style("white")     
    plt.yticks(visible=False)
    plt.legend(labels=['Solvabilité améliorée', 'Solvabilité diminuée'], loc='upper right', fontsize=15)
    return fig

def main():
    
    API_URI =  'http://erwanpgl.pythonanywhere.com/predict' #"http://127.0.0.1:5000/predict"
    
    #st.text(os.listdir()) useful for infos on server's when deployed

    st.title('Credit Solvabilité Prediction')
    
    #image = Image.open('./images/pret_a_depenser.png')
    #st.image(image)             
    st.selectbox(
        "Veuillez sélectionner le client",        
        liste_clients,
        key='id_client')    

    
    predict_btn = st.button('Prédire')

    if predict_btn and st.session_state.id_client.T != '':

        #data = [[id_client]]
        pred = None
        try:
            with st.spinner('Chargement des prédictions de solvabilité du client...'):

                pred = request_prediction(API_URI, st.session_state.id_client.T)#[0] #* 100000
                print(pred)

                st.session_state.API_data = json.loads(pred) 
                st.session_state['added_fig']  = None
                st.session_state['first_feature'] = None  
                st.session_state['second_feature'] = None

        except Exception as e:
            print("An exception occurred: {}".format(e)) #.args[0]
            st.markdown(e) 

    if st.session_state.API_data != '':

        API_data = st.session_state.API_data
        prediction_num = round(API_data['proba_defaut']*100,2)
        prediction = str(round(API_data['proba_defaut']*100,2))
        #message_resultat = 'Prédiction : **{0}** avec **{1}%** de risque de défaut'
        message_resultat = '{0} avec {1}% de risque de défaut'

        classe_predite = API_data['prediction']
        if classe_predite == 1:
            etat = 'client à risque'
            #message_resultat = ':red[' + message_resultat.format(etat, prediction) + ']'                    
        else:
            etat = 'client peu risqué'
            #message_resultat = ':green[' + message_resultat.format(etat, prediction) + ']'

        #st.markdown(message_resultat)

        fig = go.Figure(go.Indicator(
            mode = "number+delta+gauge",
            #gauge = {'shape': "bullet", 'bar.color': 'darkred' if classe_predite == 1 else 'darkgreen', 'threshold.line': 0.1},
            gauge = {'axis': {'visible': False}, 'bar.color': 'darkred' if classe_predite == 1 else 'darkgreen', 'threshold.thickness': 0.1},
            delta = {'reference':50, 'increasing.color':'red', 'decreasing.color':'green' },
            value = prediction_num,
            #domain = {'x': [0.1, 1], 'y': [0.2, 0.9]},
            domain = {'row': 0, 'column': 0},
            title = {'text':  message_resultat.format(etat, prediction), 'align': 'center', 'font_color': 'darkred' if classe_predite == 1 else 'darkgreen'},
            number={'font_color': 'darkred' if classe_predite == 1 else 'darkgreen'}                    
            ))
        

        st.plotly_chart(fig)
        
        #classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(id_input)]['LABELS'].values[0]
        #classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
            # (classe réelle : '+str(classe_reelle) + ')'   
        

        st.subheader("Caractéristiques ayant influencé la prédiction:")
        
        client_top_features = pd.read_json(API_data['client_top_features'])

        #Affichage des graphes    
        fig = graphes_client_top_features(client_top_features.iloc[:4,:])
        st.pyplot(fig)

        st.subheader("Analyses de caractéristiques sur la prédiction:")

        first_feature = st.selectbox(
            "Veuillez sélectionner la première caractéristique",
            client_top_features['Feature'],
            key='first_feature', 
            on_change= graphes_callback())    

        second_feature = st.selectbox(
            "Veuillez sélectionner la deuxième caractéristique",
            client_top_features['Feature'],
            key='second_feature', 
            on_change= graphes_callback())    
        
        if (st.session_state['added_fig']  != None): 
            st.pyplot(st.session_state['added_fig'] )

        st.subheader("Analyses des caractéristiques les plus importantes pour l'ensemble des clients:")

        global_top_10_features = pd.read_json(API_data['global_top_10_features'])
        fig= graphes_global_features(global_top_10_features.iloc[:5,:])
        st.pyplot(fig)

if __name__ == '__main__':
    main()
