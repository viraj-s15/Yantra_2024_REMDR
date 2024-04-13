import joblib

def prediction(pred_list):
    model = joblib.load('weights/sensor_model.pkl')
    pred = model.predict([pred_list])
    return pred
