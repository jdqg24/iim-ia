# main.py
from features.extract_features import extract_all_features

if __name__ == "__main__":
    extract_all_features(
        dataset_dir="data_v3/",
        output_csv="data_v3/features_dataset.csv",
        n_jobs=-1  # todos los cores
    )