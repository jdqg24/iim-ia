# main.py
from src.features.extract_features import extract_all_features

if __name__ == "__main__":
    extract_all_features(
        dataset_dir="data/raw",
        output_csv="data/features_dataset.csv",
        n_jobs=-1  # todos los cores
    )