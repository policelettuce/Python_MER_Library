import pandas as pd


def normalize_value(value):
    return round(2 * (value - 0.5), 4)


def normalize_dataset(input_csv_path, output_csv_path):
    # Load the dataset
    df = pd.read_csv(input_csv_path)

    # Select and rename the necessary columns
    selected_columns = ["musicId", "Valence(mean)", "Arousal(mean)"]
    df = df[selected_columns]

    # Normalize the values in the mean columns
    df["Valence(mean)"] = df["Valence(mean)"].apply(normalize_value)
    df["Arousal(mean)"] = df["Arousal(mean)"].apply(normalize_value)

    # Save the new dataset
    df.to_csv(output_csv_path, index=False)
    print(f"Normalized dataset saved to {output_csv_path}")


def main():
    input_csv_path = "dataset/pmemo_annotations.csv"
    output_csv_path = "dataset/pmemo_annotations_normalized.csv"

    normalize_dataset(input_csv_path, output_csv_path)


if __name__ == "__main__":
    main()
