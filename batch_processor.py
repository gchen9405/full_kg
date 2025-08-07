#!/usr/bin/env python3
"""
Input File Splitter for GraphRAG
Splits large input directories into manageable subdirectories based on document source.
"""

import shutil
from pathlib import Path

def split_input_by_source(input_dir, batch_size=7500):
    """Split input files into subdirectories within the input folder by document type and size."""
    input_path = Path(input_dir)
    
    # Create subdirectories within the input folder
    created_dirs = []
    
    # Group files by document prefix
    files_by_prefix = {}
    for file in input_path.glob("*.txt"):
        prefix = file.name.split('_')[0]  # e.g., PRELIMusc06
        if prefix not in files_by_prefix:
            files_by_prefix[prefix] = []
        files_by_prefix[prefix].append(file)
    
    print(f"Found {len(files_by_prefix)} document types:")
    for prefix, files in files_by_prefix.items():
        print(f"  - {prefix}: {len(files):,} files")
    
    print(f"\nOrganizing into subdirectories within {input_path.name}/ (max {batch_size:,} files each)...")
    
    for prefix, files in files_by_prefix.items():
        print(f"\nProcessing {prefix}: {len(files):,} files")
        
        if len(files) <= batch_size:
            # Small enough - create single subdirectory
            subdir = input_path / prefix
            subdir.mkdir(exist_ok=True)
            created_dirs.append(subdir)
            
            for file in files:
                shutil.move(str(file), str(subdir / file.name))
            
            print(f"  → Created {prefix}/ ({len(files):,} files)")
            
        else:
            # Too large - split into multiple subdirectories
            chunks = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
            
            for i, chunk in enumerate(chunks, 1):
                subdir = input_path / f"{prefix}_part{i}"
                subdir.mkdir(exist_ok=True)
                created_dirs.append(subdir)
                
                for file in chunk:
                    shutil.move(str(file), str(subdir / file.name))
                
                print(f"  → Created {prefix}_part{i}/ ({len(chunk):,} files)")
    
    return created_dirs

def main():
    base_dir = Path("/Users/gc/Documents/full_kg")
    input_dir = base_dir / "msgragtest" / "input"
    
    print("Input File Organizer for GraphRAG")
    print("=" * 40)
    print(f"Directory: {input_dir}")
    print(f"Total files: {len(list(input_dir.glob('*.txt'))):,}")
    
    print(f"\nThis will organize files into subdirectories within {input_dir.name}/")
    print("Example structure after organization:")
    print("  msgragtest/input/PRELIMusc06/")
    print("  msgragtest/input/PRELIMusc08/")  
    print("  msgragtest/input/PRELIMusc10_part1/")
    print("  msgragtest/input/PRELIMusc10_part2/")
    print("  etc.")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Organize the files
    created_dirs = split_input_by_source(input_dir, batch_size=7500)
    
    print(f"\n{'='*60}")
    print("ORGANIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Created {len(created_dirs)} subdirectories within {input_dir.name}/:")
    
    total_files = 0
    for subdir in created_dirs:
        file_count = len(list(subdir.glob("*.txt")))
        total_files += file_count
        print(f"  - {subdir.name}/: {file_count:,} files")
    
    print(f"\nTotal files organized: {total_files:,}")
    
    print(f"\n📋 Next steps:")
    print(f"1. To process a specific batch, copy one subdirectory's contents back to input/")
    print(f"2. Example: cp input/PRELIMusc06/* input/ && rm -rf input/PRELIMusc06/")
    print(f"3. Then run: python -m graphrag index --root msgragtest --resume")
    print(f"4. Repeat for each subdirectory")

if __name__ == "__main__":
    main()