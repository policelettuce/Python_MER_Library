import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# Ensure you have the appropriate NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load your XANEW dictionary (assuming you have it in a CSV format)
xanew_path = "dataset/lyrics/xanew_normalized.csv"
xanew_df = pd.read_csv(xanew_path)

# Rename columns for easier access
xanew_df.rename(columns={
    'Word': 'word',
    'V.Mean.Sum': 'valence',
    'A.Mean.Sum': 'arousal',
}, inplace=True)

# Create a dictionary for quick lookup of XANEW scores
xanew_dict = {}
for i, row in xanew_df.iterrows():
    word = row['word']
    xanew_dict[word] = {
        'valence': row['valence'],
        'arousal': row['arousal'],
    }


def get_xanew_scores(text):
    valence_sum = 0
    arousal_sum = 0
    dominance_sum = 0
    word_count = 0

    for word in text.split():
        word.lower()
        if word in xanew_dict:
            valence_sum += xanew_dict[word]['valence']
            arousal_sum += xanew_dict[word]['arousal']
            word_count += 1

    if word_count == 0:
        return {'valence': None, 'arousal': None}
    else:
        return {
            'valence': valence_sum / word_count,
            'arousal': arousal_sum / word_count,
        }


def process_lyrics(df):
    vader_analyzer = SentimentIntensityAnalyzer()

    vader_features = []
    xanew_features = []

    for lyrics in df['lyrics_cleaned']:
        # VADER sentiment analysis
        vader_scores = vader_analyzer.polarity_scores(lyrics)
        vader_features.append(vader_scores)

        # XANEW sentiment analysis
        xanew_scores = get_xanew_scores(lyrics)
        xanew_features.append(xanew_scores)

    # Convert lists of dictionaries to DataFrames
    vader_df = pd.DataFrame(vader_features)
    vader_df = vader_df.rename(columns=lambda x: 'vader_' + x)
    xanew_df = pd.DataFrame(xanew_features)
    xanew_df = xanew_df.rename(columns=lambda x: 'xanew_' + x)

    # Combine VADER and XANEW features with original data
    combined_df = pd.concat([df, vader_df, xanew_df], axis=1)

    # Select relevant columns for the new dataset

    final_columns = [
        'artist', 'trackname', 'vader_pos', 'vader_neg', 'vader_neu', 'vader_compound',
        'xanew_valence', 'xanew_arousal', 'y_valence', 'y_arousal'
    ]
    return combined_df[final_columns]


def analyze_column(df, column_name):
    if column_name not in df.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return

    col = df[column_name]

    positive_count = (col > 0).sum()
    negative_count = (col < 0).sum()
    min_value = col.min()
    max_value = col.max()

    print(f"Column '{column_name}' analysis:")
    print(f"Number of positive values: {positive_count}")
    print(f"Number of negative values: {negative_count}")
    print(f"Minimum value: {min_value}")
    print(f"Maximum value: {max_value}")


def normalize_valence(value):
    return value / 2.25


def normalize_arousal(value):
    return value / 3


def main():
    # Load the dataset
    input_csv_path = "dataset/lyrics/merged_cleaned_sentiment_train.csv"
    output_csv_path = "dataset/lyrics/lyrics_ML_full.csv"

    df = pd.read_csv(input_csv_path)

    # Process lyrics to generate features
    processed_df = process_lyrics(df)

    analyze_column(processed_df, 'xanew_valence')
    analyze_column(processed_df, 'xanew_arousal')

    processed_df['y_valence'] = processed_df['y_valence'].apply(normalize_valence)
    processed_df['y_arousal'] = processed_df['y_arousal'].apply(normalize_arousal)

    # Save the new dataset
    processed_df.to_csv(output_csv_path, index=False)
    print(f"Processed dataset saved to {output_csv_path}")



if __name__ == "__main__":
    main()
