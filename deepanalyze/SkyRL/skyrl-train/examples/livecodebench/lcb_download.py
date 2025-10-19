import os
import gdown
import shutil
import argparse

# Define the Google Drive file IDs for the JSON files
FILE_IDS = {
    "train_livecodebench.json": "1-lKdRfRjytdTltgLyAxTqVRoksI2cJfU",
    "test_livecodebench.json": "1B0sotl48BLd4gqlitL5HVJf1cy3RxpEV",
}


def main():
    parser = argparse.ArgumentParser(
        description="Download LCB dataset from Google Drive"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Base destination directory to store downloaded JSON files",
    )
    args = parser.parse_args()

    dest_dir = os.path.expanduser(args.local_dir)

    # Ensure destination subdirectories exist
    for split in ["train", "test"]:
        path = os.path.join(dest_dir, split, "code")
        os.makedirs(path, exist_ok=True)

    # Download and move each file
    for filename, file_id in FILE_IDS.items():
        split = "train" if "train" in filename else "test"
        dest_dir_path = os.path.join(dest_dir, split, "code")
        temp_path = os.path.join(
            dest_dir_path, f"temp_{filename}"
        )  # Temporary download location in dest directory
        dest_path = os.path.join(dest_dir_path, "livecodebench.json")

        print(f"Downloading {filename}...")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", temp_path, quiet=False
        )

        print(f"Moving {filename} to {dest_path}...")
        shutil.move(temp_path, dest_path)

    print("All files downloaded and moved successfully.")


if __name__ == "__main__":
    main()
