import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from params import pred_lyrics_path, xanew_dict_path, pred_lyrics_features_path


def get_lyrics_features():
    # Ensure you have the appropriate NLTK data
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    # Load the XANEW dictionary
    xanew_df = pd.read_csv(xanew_dict_path).rename(columns={
        'Word': 'word',
        'V.Mean.Sum': 'valence',
        'A.Mean.Sum': 'arousal'
    })

    # Create a dictionary for quick lookup of XANEW scores
    xanew_dict = {row['word']: {'valence': row['valence'], 'arousal': row['arousal']}
                  for _, row in xanew_df.iterrows()}

    def get_xanew_scores(text):
        valence_sum = arousal_sum = word_count = 0
        for word in text.split():
            word = word.lower()
            if word in xanew_dict:
                valence_sum += xanew_dict[word]['valence']
                arousal_sum += xanew_dict[word]['arousal']
                word_count += 1
        if word_count == 0:
            return {'valence': None, 'arousal': None}
        else:
            return {'valence': round(valence_sum / word_count, 4), 'arousal': round(arousal_sum / word_count, 4)}

    # Read the text from the .txt file
    with open(pred_lyrics_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # VADER sentiment analysis
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_scores = {f"vader_{k}": v for k, v in vader_scores.items()}

    # XANEW sentiment analysis
    xanew_scores = get_xanew_scores(text)
    xanew_scores = {f"xanew_{k}": v for k, v in xanew_scores.items()}

    # Normalize the y_valence and y_arousal
    features = {**vader_scores, **xanew_scores}

    # Create a DataFrame with a single row and save as a .csv file
    pd.DataFrame([features]).to_csv(pred_lyrics_features_path, index=False)
