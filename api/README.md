# Api de prédiction de solvabilité client

## Description
Api receiving id of client through Post hhtp and returning predictions on customer solvability. Binary classification


## Call Api - Input
 - id of client set to 'client_id' in Json dictionnary 
 - Json send in a Post http verb

## Call Api - Output
 - Json dictionnary with following keys:
 - 'prediction' : returns an int with values  0 (solvable) or 1 (not solvable)
 - 'proba_defaut' : returns a float 2 dimeansions array with exact probabilities for case 0 (solvable) and case 1 (not solvable)
 - 'features_values' : json list of all the features and their data for the concerned client (can be used for explanation)

## Compatible Configurations
- Python 3.11+
- Windows 10, Ubuntu 20, macOS

## Installation
1. Clone the repository.
2. Install required packages: `pip install -r requirements.txt`.

## Necessary packages
- Flask
- Lightgbm

## Main files
- flask_api.py: entry point, responds to post http request
- modeles/predictions.py : calculates predictions, loads model, calls feature engineering
- modeles/preprocesser.py : in charge of feature engineering
- model_lightgbm1.pkl

## Features
- Feature 1: Returns predictions for a client. Binary classification and probabilities 
- Feature 2: Returns all the features and their data engineered for the concerned client

## Getting Started (locally)
- Run `python flask_api.py`.
