import pandas as pd
from params import lyrics_model_valence_path, lyrics_model_arousal_path, lyrics_scaler_path, pred_lyrics_features_path
import joblib


def predict_lyrics():
    svr_valence = joblib.load(lyrics_model_valence_path)
    svr_arousal = joblib.load(lyrics_model_arousal_path)
    scaler = joblib.load(lyrics_scaler_path)

    data = pd.read_csv(pred_lyrics_features_path)
    feature_columns = ['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound', 'xanew_valence', 'xanew_arousal']
    data = data[feature_columns]

    X_new = scaler.transform(data)

    valence_pred = svr_valence.predict(X_new)
    arousal_pred = svr_arousal.predict(X_new)

    # print(f'Predicted Valence: {valence_pred[0]:.3f}')
    # print(f'Predicted Arousal: {arousal_pred[0]:.3f}')

    return [valence_pred[0], arousal_pred[0]]
