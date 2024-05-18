import pandas as pd
from params import lyrics_model_path, lyrics_scaler_path, pred_lyrics_features_path
import joblib


def predict_lyrics():
    svr_model = joblib.load(lyrics_model_path)
    scaler = joblib.load(lyrics_scaler_path)

    data = pd.read_csv(pred_lyrics_features_path)
    feature_columns = ['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound', 'xanew_valence', 'xanew_arousal']
    data = data[feature_columns]

    X_new = scaler.transform(data)

    pred = svr_model.predict(X_new)
    pred = pred[0]

    # print(f'Predicted Valence: {valence_pred[0]:.3f}')
    # print(f'Predicted Arousal: {arousal_pred[0]:.3f}')

    return [pred[0], pred[1]]
