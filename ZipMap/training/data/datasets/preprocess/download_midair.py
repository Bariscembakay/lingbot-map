#!/usr/bin/env python3

import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import threading
from pathlib import Path

def create_directories_from_url(url, base_dir="."):
    """Create directory structure based on URL path, handling naming conflicts."""
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p and p != 'MidAir']
    
    current_path = Path(base_dir)
    for part in path_parts[:-1]:  # Exclude the filename
        current_path = current_path / part
        
        # Handle potential file/folder conflicts
        if current_path.exists() and current_path.is_file():
            # If a file exists with the same name, rename it
            backup_name = current_path.with_suffix(current_path.suffix + '.backup')
            current_path.rename(backup_name)
            print(f"⚠️  Renamed conflicting file: {current_path} -> {backup_name}")
        
        current_path.mkdir(exist_ok=True)
    
    return current_path

def download_file(url, session=None, lock=None):
    """Download a single file with progress indication."""
    if session is None:
        session = requests.Session()
    
    try:
        parsed_url = urlparse(url)
        filename = parsed_url.path.split('/')[-1]
        if '?' in filename:
            filename = filename.split('?')[0]
        
        # Create directory structure
        target_dir = create_directories_from_url(url)
        filepath = target_dir / filename
        
        # Skip if file already exists (compare with remote size if available)
        if filepath.exists():
            local_size = filepath.stat().st_size
            remote_size = 0
            try:
                head = session.head(url, allow_redirects=True, timeout=15)
                remote_size = int(head.headers.get('content-length', 0))
            except Exception:
                pass

            if local_size > 0 and (remote_size == 0 or local_size == remote_size):
                with lock:
                    size_mb = local_size / (1024 * 1024)
                    print(f"⏭️  Skipping (exists, {size_mb:.1f} MB): {filepath}")
                return True
            else:
                # Size mismatch or zero-length file -> re-download
                with lock:
                    if remote_size and local_size != remote_size:
                        print(f"♻️  Re-downloading (size mismatch {local_size} != {remote_size} bytes): {filepath}")
                    else:
                        print(f"♻️  Re-downloading: {filepath}")
                try:
                    if local_size > 0:
                        filepath.unlink()
                except FileNotFoundError:
                    pass
        
        # Download the file
        with lock:
            print(f"🔽 Starting: {filepath}")
        
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        with lock:
            if total_size > 0:
                size_mb = total_size / (1024 * 1024)
                print(f"✅ Completed: {filepath} ({size_mb:.1f} MB)")
            else:
                print(f"✅ Completed: {filepath}")
        
        return True
        
    except Exception as e:
        with lock:
            print(f"❌ Failed: {filepath} - {str(e)}")
        return False

def main():
    config_file = "download_config.txt"
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    
    if not os.path.exists(config_file):
        print(f"Error: {config_file} not found!")
        return 1
    
    # Read URLs from config file
    with open(config_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    print(f"Starting parallel downloads with {max_workers} workers")
    print(f"Total URLs: {len(urls)}")
    print("-" * 50)
    
    # Create a session for connection pooling
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    })
    
    # Thread lock for synchronized printing
    print_lock = threading.Lock()
    
    successful_downloads = 0
    failed_downloads = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_file, url, session, print_lock): url 
            for url in urls
        }
        
        # Process completed downloads
        for future in as_completed(future_to_url):
            if future.result():
                successful_downloads += 1
            else:
                failed_downloads += 1
    
    print("-" * 50)
    print(f"📊 Download Summary:")
    print(f"   ✅ Successful: {successful_downloads}")
    print(f"   ❌ Failed: {failed_downloads}")
    print(f"   📁 Total: {len(urls)}")
    
    return 0 if failed_downloads == 0 else 1

if __name__ == "__main__":
    sys.exit(main())