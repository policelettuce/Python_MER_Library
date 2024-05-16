import pandas as pd


def normalize_value_valence(value):
    return (value - 5.86) / 4


def normalize_value_arousal(value):
    return (value - 4.075) / 4


def normalize_dataset(input_csv_path, output_csv_path):
    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Select and rename the necessary columns
    selected_columns = ["Word", "V.Mean.Sum", "A.Mean.Sum"]
    df = df[selected_columns]

    # Normalize the values in the mean columns
    df["V.Mean.Sum"] = df["V.Mean.Sum"].apply(normalize_value_valence)
    df["A.Mean.Sum"] = df["A.Mean.Sum"].apply(normalize_value_arousal)

    # Save the new dataset
    df.to_csv(output_csv_path, index=False)
    print(f"Normalized dataset saved to {output_csv_path}")


def main():
    input_csv_path = "dataset/lyrics/Ratings_Warriner_et_al.csv"
    output_csv_path = "dataset/lyrics/xanew_normalized.csv"

    normalize_dataset(input_csv_path, output_csv_path)


if __name__ == "__main__":
    main()
