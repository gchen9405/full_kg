import os
import subprocess
import sys
from typing import List
from io import StringIO
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter

WORKSPACE = "./msgragtest"
INPUT_DIR = os.path.join(WORKSPACE, "input")
QUERY = "What does the US Coast Guard do?"


# --- Initialize workspace (if not already initialized) ---
if not (os.path.exists(os.path.join(WORKSPACE, ".env")) and os.path.exists(os.path.join(WORKSPACE, "settings.yaml"))):
    print("[INFO] Initializing GraphRAG workspace...")
    subprocess.run([sys.executable, "-m", "graphrag", "init", "--root", WORKSPACE], check=True)
else:
    print("[INFO] GraphRAG workspace already initialized.")


# --- Auto Tuning Config ---
AUTO_TUNING_CONFIG = {
    "domain": "US Legal Code",  # You can customize this based on your content
    "selection_method": "auto",   # Options: random, top, all, auto
    "limit": 50,                   # Number of text units to use for prompt generation
    "n_subset_max": 1000,
    "k": 50,
    "chunk_size": 512,
    "max_tokens": 4096,
    "min_examples_required": 5,
    "language": "English",         # Language for prompt tuning
    "discover_entity_types": True,  # Set to True if you want automatic entity discovery
    "output": "prompts"             # Output directory for generated prompts
}

def run_auto_prompt_tuning(workspace_path: str, config_path: str = None):
    """
    Run GraphRAG auto prompt tuning to create domain-adapted prompts.
    This should be run after workspace initialization but before indexing.
    """
    print("[INFO] Running auto prompt tuning...")
    
    # Determine config path if not provided
    if config_path is None:
        config_path = os.path.join(workspace_path, "settings.yaml")
    
    # Build the command with all parameters
    cmd = [
        sys.executable, "-m", "graphrag", "prompt-tune",
        "--root", workspace_path,
        "--config", config_path,
        "--domain", AUTO_TUNING_CONFIG["domain"],
        "--selection-method", AUTO_TUNING_CONFIG["selection_method"],
        "--limit", str(AUTO_TUNING_CONFIG["limit"]),
        "--language", AUTO_TUNING_CONFIG["language"],
        "--max-tokens", str(AUTO_TUNING_CONFIG["max_tokens"]),
        "--chunk-size", str(AUTO_TUNING_CONFIG["chunk_size"]),
        "--min-examples-required", str(AUTO_TUNING_CONFIG["min_examples_required"]),
        "--output", AUTO_TUNING_CONFIG["output"]
    ]
    
    # Add discover entity types flag if enabled
    if AUTO_TUNING_CONFIG["discover_entity_types"]:
        cmd.append("--discover-entity-types")
    else:
        cmd.append("--no-discover-entity-types")
    
    try:
        # Set environment variable to disable numba caching to avoid hyppo library issues
        env = os.environ.copy()
        env["NUMBA_DISABLE_JIT"] = "1"
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        print("[SUCCESS] Auto prompt tuning completed successfully!")
        print(f"[INFO] Generated prompts saved to: {os.path.join(workspace_path, AUTO_TUNING_CONFIG['output'])}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Auto prompt tuning failed: {e}")
        print(f"[ERROR] stderr: {e.stderr}")
        return False



def parse_html_to_chunks(html_path: str, headers_to_split_on=None, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Parse an HTML file into structured chunks using HTMLSectionSplitter and further split with RecursiveCharacterTextSplitter.
    """
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("h4", "Header 4"),
        ]
    splitter = HTMLSectionSplitter(headers_to_split_on)
    
    # Try multiple encodings to handle different file formats
    encodings = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1']
    html_str = None
    
    for encoding in encodings:
        try:
            with open(html_path, "r", encoding=encoding) as f:
                html_str = f.read()
            print(f"[DEBUG] Successfully read {html_path} with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    
    if html_str is None:
        raise ValueError(f"Could not decode {html_path} with any of the attempted encodings: {encodings}")
    
    html_buffer = StringIO(html_str)
    html_splits = splitter.split_text_from_file(html_buffer)
    # Further split large sections if needed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(html_splits)
    all_chunks = [chunk.page_content for chunk in chunks]
    return all_chunks

# --- Ensure workspace and input exist ---
os.makedirs(INPUT_DIR, exist_ok=True)


"""

print("[INFO] Generating chunks from HTML files in ./titles directory...")
titles_dir = "./titles"
total_chunks = 0

for filename in os.listdir(titles_dir):
    if filename.endswith('.htm'):
        file_path = os.path.join(titles_dir, filename)
        file_base = os.path.splitext(filename)[0]  # Remove .htm extension
        
        print(f"[INFO] Processing {filename}...")
        
        try:
            chunks = parse_html_to_chunks(file_path)
            
            for i, chunk in enumerate(chunks):
                path = os.path.join(INPUT_DIR, f"{file_base}_chunk_{i:03d}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(chunk)
            
            print(f"[INFO] Wrote {len(chunks)} chunks from {filename}")
            total_chunks += len(chunks)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            print(f"[INFO] Continuing with next file...")
            continue

print(f"[INFO] Total: {total_chunks} chunk files written to {INPUT_DIR}")

"""

# --- Run auto prompt tuning ---
print("[INFO] Running auto prompt tuning for domain-adapted prompts...")
tuning_success = run_auto_prompt_tuning(WORKSPACE)

if tuning_success:
    print("[INFO] Auto-tuning completed successfully - GraphRAG will use the tuned prompts automatically")
else:
    print("[WARNING] Auto prompt tuning failed, proceeding with default prompts")

# --- Index the input file ---
# Check if indexing is already complete by looking for output files
output_dir = os.path.join(WORKSPACE, "output")

print("[INFO] Indexing input file with GraphRAG...")
# Set environment variable to disable numba caching to avoid hyppo library issues
env = os.environ.copy()
env["NUMBA_DISABLE_JIT"] = "1"
subprocess.run([sys.executable, "-m", "graphrag", "index", "--root", WORKSPACE], check=True, env=env)

# --- Run a query ---
print("[INFO] Running a query with GraphRAG...")
try:
    result = subprocess.run([
        sys.executable, "-m", "graphrag", "query",
        "--root", WORKSPACE,
        "--method", "local",
        "--query", QUERY
    ], capture_output=True, text=True, check=True)

except subprocess.CalledProcessError as e:
    print("❌ Graphrag failed!")
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)   # ← here’s where the rate‑limit or 401 messages will live
    raise

print("\n================= QUERY RESULT =================")
print(result.stdout)
print("================================================\n") 
