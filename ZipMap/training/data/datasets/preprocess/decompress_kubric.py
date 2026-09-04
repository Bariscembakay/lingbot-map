import os
import tarfile
import multiprocessing
from functools import partial
from tqdm import tqdm

# --- Configuration ---
# 1. Directory containing your .tar.gz files
SOURCE_DIR = "/home/haian/.cache/huggingface/hub/datasets--facebook--CoTracker3_Kubric/snapshots/a23b6e6e332595c9d92944680a10415e93bfd1e1/"  # Use "." for the current directory

# 2. Directory where you want to extract the files
TARGET_DIR = "/home/storage-bucket/haian/zipmap_data/processed_cotracker3_kubric"

# 3. Number of parallel processes to use (None means use all available CPU cores)
NUM_PROCESSES = None
# ---------------------


def decompress_file(file_path, target_directory):
    """Worker function to decompress a single .tar.gz file."""
    try:
        file_name = os.path.basename(file_path)
        # print(f"Extracting {file_name}...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=target_directory)
        return f"Successfully extracted {file_name}"
    except Exception as e:
        return f"Error extracting {os.path.basename(file_path)}: {e}"


if __name__ == "__main__":
    # This check is essential for multiprocessing to work correctly
    
    # 1. Create the target directory if it doesn't exist
    print(f"📁 Creating target directory: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 2. Find all .tar.gz files in the source directory
    try:
        all_files = os.listdir(SOURCE_DIR)
        tar_files = [
            os.path.join(SOURCE_DIR, f)
            for f in all_files
            if f.endswith(".tar.gz")
        ]
    except FileNotFoundError:
        print(f"❌ Error: Source directory not found at '{SOURCE_DIR}'")
        exit()

    if not tar_files:
        print("🤔 No .tar.gz files found to extract.")
        exit()

    print(f"Found {len(tar_files)} files to decompress.")

    # 3. Set up and run the multiprocessing pool
    # Use all available cores if NUM_PROCESSES is not set
    cpu_count = os.cpu_count()
    processes_to_use = NUM_PROCESSES if NUM_PROCESSES is not None else cpu_count
    print(f"🚀 Starting decompression with {processes_to_use} processes...")

    # `partial` is used to "pre-fill" the target_directory argument for our worker function
    worker_func = partial(decompress_file, target_directory=TARGET_DIR)

    with multiprocessing.Pool(processes=processes_to_use) as pool:
        # Use tqdm for a nice progress bar
        # pool.imap_unordered is efficient as it doesn't wait for tasks to finish in order
        results = list(
            tqdm(pool.imap_unordered(worker_func, tar_files), total=len(tar_files))
        )
    
    print("\n✅ All tasks complete.")
    
    # Optional: Print any errors that occurred
    # for res in results:
    #     if "Error" in res:
    #         print(res)