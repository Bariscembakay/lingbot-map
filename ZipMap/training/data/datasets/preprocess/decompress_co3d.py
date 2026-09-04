import os
import zipfile
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tqdm


def unzip_file(zip_path, extract_to):
    """Unzip a single file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                try:
                    zip_ref.extract(member, extract_to)
                except FileExistsError:
                    if not member.endswith('/'):
                        zip_ref.extract(member, extract_to)
        return f"Successfully extracted {zip_path}"
    except Exception as e:
        return f"Error extracting {zip_path}: {str(e)}"


def find_zip_files(data_dir):
    """Find all zip files in the data directory"""
    zip_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append(os.path.join(root, file))
    return zip_files


def main():
    parser = argparse.ArgumentParser(description='Unzip CO3D dataset files in parallel')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Directory containing CO3D zip files')
    parser.add_argument('--extract_to', type=str, default=None,
                       help='Directory to extract files to (default: same as data_dir)')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(),
                       help='Number of parallel workers')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    extract_to = Path(args.extract_to) if args.extract_to else data_dir
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    print(f"Finding zip files in {data_dir}...")
    zip_files = find_zip_files(data_dir)
    
    if not zip_files:
        print("No zip files found!")
        return
    
    print(f"Found {len(zip_files)} zip files")
    print(f"Using {args.num_workers} workers")
    print(f"Extracting to: {extract_to}")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for zip_path in zip_files:
            future = executor.submit(unzip_file, zip_path, extract_to)
            futures.append(future)
        
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            result = future.result()
            if result.startswith("Error"):
                print(result)


if __name__ == "__main__":
    main()
