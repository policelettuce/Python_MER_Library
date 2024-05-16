from wrapper_predict_lyrics import predict_lyrics
from wrapper_preprocess_audio import preprocess_audio
from wrapper_predict_audio import predict_audio
from wrapper_preprocess_lyrics import get_lyrics
from wrapper_get_labels_from_emotion_space import get_labels


def predict(audiofile_path, artist_name=None, track_title=None, lyrics_score_weight=None, do_not_analyze_lyrics=False):
    preprocess_audio(audiofile_path)
    audio_va_score = predict_audio()
    valence_audio = audio_va_score[0][0]
    arousal_audio = audio_va_score[0][1]
    # Check if user specified to not search/analyze lyrics
    if not do_not_analyze_lyrics:
        if artist_name is None or track_title is None:
            raise ValueError("Artist name and track title must be provided if do_not_analyze_lyrics is not set.")

        lyrics_search_result = get_lyrics(artist_name, track_title)
        # Check if lyrics are found successfully
        if lyrics_search_result == -1:
            print("Lyrics analysis failed, analyzing audio only...")
            print(f"- - - RESULTING VA SCORE - - -\nValence: {valence_audio:.3f}\nArousal: {arousal_audio:.3f}")
            labels = get_labels(valence=valence_audio, arousal=arousal_audio)
            return labels
        else:
            valence_lyrics, arousal_lyrics = predict_lyrics()

        # Check if weights are provided, if not - set default
        if lyrics_score_weight is None:
            lyrics_score_weight = 0.5
            audio_score_weight = 0.5
        else:
            audio_score_weight = 1 - lyrics_score_weight

        # Ensemble VA Scores with provided weights
        valence_ensembled = valence_lyrics * lyrics_score_weight + valence_audio * audio_score_weight
        arousal_ensembled = arousal_lyrics * lyrics_score_weight + arousal_audio * audio_score_weight
        print(f"- - - RESULTING VA SCORE - - -\nValence: {valence_ensembled:.3f}\nArousal: {arousal_ensembled:.3f}")
        labels = get_labels(valence=valence_ensembled, arousal=arousal_ensembled)
        return labels
    else:
        print(f"- - - RESULTING VA SCORE - - -\nValence: {valence_audio:.3f}\nArousal: {arousal_audio:.3f}")
        labels = get_labels(valence=valence_audio, arousal=arousal_audio)
        return labels


if __name__ == '__main__':
    audio_path = "benjamin.mp3"
    artist = "Kanye West"
    title = "On Sight"

    labels = predict(audiofile_path=audio_path, artist_name=artist, track_title=title, do_not_analyze_lyrics=True)
    print(f"Resulting labels are: {labels}")
