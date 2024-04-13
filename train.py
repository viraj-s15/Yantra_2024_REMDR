
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import joblib
import optuna
from lightgbm import LGBMRegressor
df = pd.read_csv('dataset.csv')

X = df.drop('life_probability', axis=1)
y = df['life_probability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        'max_bin': trial.suggest_int('max_bin', 100, 500),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 0.1),
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = ((y_pred - y_test) ** 2).mean() ** 0.5
    return -rmse

study = optuna.create_study(direction='minimize')  
study.optimize(objective, n_trials=100)

best_params = study.best_params
final_model = LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)


joblib.dump(final_model, 'weights/sensor_model.pkl') 
