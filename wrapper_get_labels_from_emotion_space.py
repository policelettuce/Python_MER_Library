from EMOTION_SPACE_MAP import EmotionSpaceMap as map


def get_labels(valence, arousal):
    if valence <= -0.5 and arousal >= 0.5:
        return map.Q1_1
    elif -0.5 <= valence <= 0 and arousal >= 0.5:
        return map.Q1_2
    elif valence <= -0.5 and 0.5 >= arousal >= 0:
        return map.Q1_3
    elif -0.5 <= valence <= 0 and 0.5 >= arousal >= 0:
        return map.Q1_4

    elif 0.5 >= valence >= 0 and arousal >= 0.5:
        return map.Q2_1
    elif valence >= 0.5 and arousal >= 0.5:
        return map.Q2_2
    elif 0.5 >= valence >= 0 and 0.5 >= arousal >= 0:
        return map.Q2_3
    elif valence >= 0.5 and 0.5 >= arousal >= 0:
        return map.Q2_4

    elif valence <= -0.5 and 0 >= arousal >= -0.5:
        return map.Q3_1
    elif 0 >= valence >= -0.5 and 0 >= arousal >= -0.5:
        return map.Q3_2
    elif valence <= -0.5 and arousal <= -0.5:
        return map.Q3_3
    elif 0 >= valence >= -0.5 and arousal <= -0.5:
        return map.Q3_4

    elif 0.5 >= valence >= 0 and 0 >= arousal >= -0.5:
        return map.Q4_1
    elif valence >= 0.5 and 0 >= arousal >= -0.5:
        return map.Q4_2
    elif 0.5 >= valence >= 0 and arousal <= -0.5:
        return map.Q4_3
    elif valence >= 0.5 and arousal <= -0.5:
        return map.Q4_4

    else:
        raise ValueError("Out of bounds VA Score!")
