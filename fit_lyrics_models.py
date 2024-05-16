import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from params import lyrics_model_valence_path, lyrics_model_arousal_path, lyrics_scaler_path


df = pd.read_csv('dataset/lyrics/lyrics_ML_full.csv')
df = df.dropna()

X = df[['vader_pos', 'vader_neg', 'vader_neu', 'vader_compound', 'xanew_valence', 'xanew_arousal']]
y_valence = df['y_valence']
y_arousal = df['y_arousal']

X_train, X_test, y_arousal_train, y_arousal_test = train_test_split(X, y_arousal, test_size=0.2, random_state=42)
X_train, X_test, y_valence_train, y_valence_test = train_test_split(X, y_valence, test_size=0.2, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVR model for valence
svr_valence = SVR(kernel='rbf')
svr_valence.fit(X_train, y_valence_train)

# Train the SVR model for arousal
svr_arousal = SVR(kernel='rbf')
svr_arousal.fit(X_train, y_arousal_train)

# Predict and evaluate
y_valence_pred = svr_valence.predict(X_test)
valence_mse = mean_squared_error(y_valence_test, y_valence_pred)
valence_r2 = r2_score(y_valence_test, y_valence_pred)

print(f'Valence MSE: {valence_mse:.3f}')
print(f'Valence R2: {valence_r2:.3f}')

y_arousal_pred = svr_arousal.predict(X_test)
arousal_mse = mean_squared_error(y_arousal_test, y_arousal_pred)
arousal_r2 = r2_score(y_arousal_test, y_arousal_pred)

print(f'Arousal MSE: {arousal_mse:.3f}')
print(f'Arousal R2: {arousal_r2:.3f}')

# Save the trained SVR models and scaler
joblib.dump(svr_valence, lyrics_model_valence_path)
joblib.dump(svr_arousal, lyrics_model_arousal_path)
joblib.dump(scaler, lyrics_scaler_path)
