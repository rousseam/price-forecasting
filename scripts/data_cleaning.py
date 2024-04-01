import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

X = pd.read_csv('data_analysis/raw_data/X_train_Wwou3IE.csv')
y = pd.read_csv('data_analysis/raw_data/y_train_jJtXgMX.csv')
X_rendu = pd.read_csv('data_analysis/raw_data/X_test_GgyECq8.csv')

def to_seconds(hour_stamp):
    return int(hour_stamp.split(":")[0])*60*60

def date_stamp(df):
    df["year"] = df["DELIVERY_START"].apply(lambda x: x.split("-")[0])
    df["month"] = df["DELIVERY_START"].apply(lambda x: x.split("-")[1])
    df["day"] = df["DELIVERY_START"].apply(lambda x: (x.split("-")[2]).split(" ")[0])
    df["seconds"] = df["DELIVERY_START"].apply(lambda x: to_seconds(x.split(" ")[1]))
    return df
    

X = date_stamp(X)
X_rendu  = date_stamp(X_rendu)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#On spécifie dans quelle colonne on va chercher les valeurs manquantes (On perd 34 lignes)

X_train_kept = X_train.dropna(subset=['coal_power_available', 'gas_power_available', 'nucelear_power_available', 'wind_power_forecasts_average', 'solar_power_forecasts_average', 'wind_power_forecasts_std', 'solar_power_forecasts_std'])
X_rendu_kept = X_train.dropna(subset=['coal_power_available', 'gas_power_available', 'nucelear_power_available', 'wind_power_forecasts_average', 'solar_power_forecasts_average', 'wind_power_forecasts_std', 'solar_power_forecasts_std'])

marked_train = pd.concat([X_train, X_train_kept]).drop_duplicates(keep=False)['DELIVERY_START']
marked_rendu = pd.concat([X_rendu, X_rendu_kept]).drop_duplicates(keep=False)['DELIVERY_START']


X_train = X_train[~X_train['DELIVERY_START'].isin(marked_train)]
y_train = y_train[~y_train['DELIVERY_START'].isin(marked_train)]
X_rendu = X_rendu[~X_rendu['DELIVERY_START'].isin(marked_train)]

y_train.to_csv('data_analysis/data/y_train2.csv')

#Représentation graphique 

#X_test.hist(figsize=(16, 9), bins=50, xlabelsize=8, ylabelsize=8)
#plt.show()

#sns.distplot(y_train['spot_id_delta'], color='g', bins=1000, hist_kws={'alpha': 0.4})
#plt.show()

threshold = 600

eliminated_starts = y_train[abs(y_train['spot_id_delta']) - threshold >= 0].DELIVERY_START

#print(eliminated_starts) #On enlève 3 valeurs

y_train = y_train[~y_train['DELIVERY_START'].isin(eliminated_starts)] # on ne sélectionne que les valeurs qui ne correspondent pas aux dates enlevées
X_train = X_train[~X_train['DELIVERY_START'].isin(eliminated_starts)] # de même ici pour être cohérent sur le nombre de lignes

#Prédiction de load_forecast

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler

#Données normales
y_load = X_train.dropna(subset=['load_forecast'])['load_forecast']
X_load = X_train.dropna(subset=['load_forecast'])
X_load = X_load.loc[:, X_load.columns != 'predicted_spot_price']
X_load = X_load.loc[:, X_load.columns != 'load_forecast']

X_train_load, X_test_load, y_train_load, y_test_load = train_test_split(X_load, y_load, test_size=0.3)

# Données normalisées 
X_train_scaled = X_train_load.drop(columns=['DELIVERY_START'])
X_test_scaled = X_test_load.drop(columns=['DELIVERY_START'])
scaler = StandardScaler()
scaler.fit(X_train_scaled)
X_train_scaled = scaler.transform(X_train_scaled)
X_test_scaled = scaler.transform(X_test_scaled)

#clf = LinearRegression()
#lf.fit(X_train_load.set_index('DELIVERY_START'), y_train_load)
#print('linear regression score: ', clf.score(X_test_load.set_index('DELIVERY_START'), y_test_load))

#clf = GradientBoostingRegressor()
#clf.fit(X_train_load.set_index('DELIVERY_START'), y_train_load)
#print('gradient boosting regression score: ', clf.score(X_test_load.set_index('DELIVERY_START'), y_test_load))

clf = RandomForestRegressor(n_estimators=150, criterion= 'poisson') #Poisson et squared sont les mieux
#MAE n'a pas de sens car on a déjà enlevé les valeurs abérrantes
clf.fit(X_train_load.set_index('DELIVERY_START'), y_train_load)
print('random forest regression score: ', clf.score(X_test_load.set_index('DELIVERY_START'), y_test_load))

#clf_elastic_net = ElasticNet()
#clf_elastic_net.fit(X_train_load.set_index('DELIVERY_START'), y_train_load)
#print('ElasticNet regression score: ', clf_elastic_net.score(X_test_load.set_index('DELIVERY_START'), y_test_load))

#Cross validation

from sklearn.model_selection import cross_val_score

X_load_cv = X_load.drop(columns=['DELIVERY_START'])
y_load_cv = y_load

# Mise à l'échelle des caractéristiques pour la validation croisée (inutile pour la random forest)
#scaler = StandardScaler()
#X_load_cv_scaled = scaler.fit_transform(X_load_cv)

# Initialisation du modèle RandomForestRegressor
clf_cv = RandomForestRegressor(n_estimators=150, criterion='poisson')
scores = cross_val_score(clf_cv, X_load_cv, y_load_cv, cv=5, scoring='r2') #On fait 5

# Affichage des résultats
print("R2 scores from cross-validation: ", scores) #Verif overfitting
print("Mean R2 score: ", scores.mean()) #Évite les scores chanceux/malchanceux
print("Standard deviation of R2 scores: ", scores.std()) #Faible indique que le modèle est stable peu importe les données

#Les scores sont globalement toujours les mêmes sauf pour l'écart type. 
#Les scores d'écart type dépendent surtout du nombre de branche. Le mieux semble être entre 100 et 175. 
#On remarque aussi que l'écart type est plus faire pour le modèle de GradientBoostingRegression
#La différence n'est cependant pas suffisante pour préférer ce dernier à RandomForestRegression


#On choisit donc RandomForestRegression

X_predict_load = X_train[X_train['load_forecast'].isna()]
predicted_spot_price1 = X_predict_load['predicted_spot_price']
load_forecast1 = X_predict_load['load_forecast']
X_predict_load = X_predict_load.loc[:, ~(X_predict_load.columns.isin(['predicted_spot_price', 'load_forecast']))]

X_predict_load.insert(1, 'load_forecast', clf.predict(X_predict_load.set_index('DELIVERY_START')))
X_predict_load.insert(9, 'predicted_spot_price', predicted_spot_price1)

X_train = pd.concat([X_train[~X_train['load_forecast'].isna()], X_predict_load])

#Prédiction de predicted_spot_price

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint

delivery_starts = X_train['DELIVERY_START']

y_spot = X_train.dropna(subset=['predicted_spot_price'])['predicted_spot_price']
X_spot = X_train.dropna(subset=['predicted_spot_price'])
X_spot = X_spot.loc[:, X_spot.columns != 'predicted_spot_price']
X_spot = X_spot.set_index('DELIVERY_START')

print('nombre de valeurs non NaN: ', len(y_spot))

X_train_spot, X_test_spot, y_train_spot, y_test_spot = train_test_split(X_spot, y_spot, test_size=0.5)

clf_lr = LinearRegression()
clf_lr.fit(X_train_spot, y_train_spot)
print('linear regression score: ', clf_lr.score(X_test_spot, y_test_spot))

clf_rf = RandomForestRegressor(n_estimators=200)
clf_rf.fit(X_train_spot, y_train_spot)
print('random forest regression score: ', clf_rf.score(X_test_spot, y_test_spot))

clf_gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1)
clf_gb.fit(X_train_spot, y_train_spot)
print('gradient boosting regression score: ', clf_gb.score(X_test_spot, y_test_spot))

scores_lr = cross_val_score(clf_lr, X_spot, y_spot, cv=5, scoring='r2')
scores_rf = cross_val_score(clf_rf, X_spot, y_spot, cv=5, scoring='r2')
scores_gb = cross_val_score(clf_gb, X_spot, y_spot, cv=5, scoring='r2')

print("R2 scores from cross-validation: ", scores_lr, scores_rf, scores_gb) #Verif overfitting
print("Mean R2 score: ", scores_lr.mean(), scores_rf.mean(), scores_gb.mean()) #Évite les scores chanceux/malchanceux
print("Standard deviation of R2 scores: ", scores_lr.std(), scores_rf.std(), scores_gb.std()) #Faible indique que le modèle est stable peu importe les données

#GradientBoosting est meilleur en moyenne et écart-type, on fait monter la moyenne en augmentant le nombre d'estimateur. 
#Il faut cependant faire attention à l'overfitting car augmenter le nombre d'estimateur
#augmente grandement l'écart-type.
#Nous avons essayé de toucher au learning_rate mais la valeur la plus efficace est celle fixée par défaut

X_predict_spot = X_train[X_train['predicted_spot_price'].isna()]
X_predict_spot = X_predict_spot.loc[:, X_predict_spot.columns != 'predicted_spot_price']
X_predict_spot = X_predict_spot.set_index('DELIVERY_START')

X_predict_rendu = X_rendu[X_rendu['predicted_spot_price'].isna()]
X_predict_rendu = X_predict_rendu.loc[:, X_predict_rendu.columns != 'predicted_spot_price']
X_predict_rendu = X_predict_rendu.set_index('DELIVERY_START')

X_predict_spot.insert(8, 'predicted_spot_price', clf_gb.predict(X_predict_spot))
X_predict_rendu.insert(8, 'predicted_spot_price', clf_gb.predict(X_predict_rendu))

X_train = pd.concat([X_train[~X_train['predicted_spot_price'].isna()], X_predict_spot.reset_index('DELIVERY_START')])
X_rendu = pd.concat([X_rendu[~X_rendu['predicted_spot_price'].isna()], X_predict_rendu.reset_index('DELIVERY_START')])

X_train.to_csv('data_analysis/data/X_train2.csv')
X_rendu.to_csv('data_analysis/data/X_test2.csv')