from lyricsgenius import Genius
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import wrapper_get_lyrics_features
from params import pred_lyrics_path, xanew_dict_path


def clear_lyrics(text: str):
    lines = text.split('\n', 1)
    if len(lines) > 1:
        text = lines[1]

    text = re.sub(r'\[.*?\]', '', text)

    text = re.sub(r'\d+Embed$', '', text, flags=re.MULTILINE)

    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()

    return text


def get_lyrics(artist, title):
    gen = Genius('7K-nH_pWWuZC2S-EEpbj8k_policelettuce_Oo8KC_owajt0chBSC7uDwJxVgF_b58Wo')
    # Try to search for lyrics by user's input, if lyrics not found - skip lyrics prediction
    try:
        text = gen.search_song(artist=artist, title=title, get_full_info=True).lyrics
    except Exception:
        print("Skipping lyrics prediction...")
        return -1

    clean_text = clear_lyrics(text)
    with open(pred_lyrics_path, "w", encoding="utf-8") as file:
        file.write(clean_text)
    wrapper_get_lyrics_features.get_lyrics_features()
