import os
import shutil
import kagglehub


def fetch_dataset():
    os.makedirs("data", exist_ok=True)
    target_path = os.path.join("data", "creditcard.csv")

    if os.path.exists(target_path):
        print(f"Dataset already exists at {target_path}")
        return

    print("Downloading Credit Card Fraud dataset from Kaggle...")
    downloaded_dir = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    source_csv = os.path.join(downloaded_dir, "creditcard.csv")

    shutil.copy(source_csv, target_path)
    print(f"Saved dataset to {target_path}")


if __name__ == "__main__":
    fetch_dataset()