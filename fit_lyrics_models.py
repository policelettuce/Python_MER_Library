import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from params import lyrics_model_path, lyrics_scaler_path


df = pd.read_csv('dataset/lyrics/lyrics_ML_full.csv')
df = df.dropna()

X = df[['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound', 'xanew_valence', 'xanew_arousal']]
y = df[['y_valence', 'y_arousal']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svr = SVR(kernel='rbf')
multi_target_svr = MultiOutputRegressor(svr)
multi_target_svr.fit(X_train, y_train)

y_pred = multi_target_svr.predict(X_test)
valence_mse = mean_squared_error(y_test['y_valence'], y_pred[:, 0])
arousal_mse = mean_squared_error(y_test['y_arousal'], y_pred[:, 1])

print(f'Valence MSE: {valence_mse:.3f}')
print(f'Arousal MSE: {arousal_mse:.3f}')

joblib.dump(multi_target_svr, lyrics_model_path)
joblib.dump(scaler, lyrics_scaler_path)
