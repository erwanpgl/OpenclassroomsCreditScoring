from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pandas as pd
from modeles.predictions import predict

app = Flask(__name__)
api = Api(app)

# Function to test if the request contains multiple 
def islist(obj):
    return True if ("list" in str(type(obj))) else False

class Preds(Resource):
    def post(self):

        # Get POST data as json & read it as a DataFrame
        json_ = request.get_json()
        print (json_)
        
        id_client = int(json_['client_id'])
        
        prediction, proba, top_10_features = predict(id_client)        #json_data = json.dumps(array.tolist())

        dict_final = {
        'prediction' : int(prediction),
        'proba_defaut' : float(proba[0][1]),
        'top_10_features' : top_10_features.to_json()     
        }

        print('Nouvelle Prédiction : \n', dict_final)

        return dict_final, 200

        res = {'predictions': {}}
        # Create the response
        for i in range(len(prediction[1][0])):
          res['predictions'][i + 1] = int(prediction[1][0][i])
        return res, 200 # Send the response object
        #return {'message': 'POST data read successfully'}

         

    

          
    

api.add_resource(Preds, '/predict')

if __name__ == "__main__":
    app.run(debug = True)#, port = 80)