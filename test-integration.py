import os
import subprocess
import sys
from typing import List
from io import StringIO
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSectionSplitter, RecursiveCharacterTextSplitter

# --- Config ---
WORKSPACE = "./msgragtest"
INPUT_DIR = os.path.join(WORKSPACE, "input")
QUERY = "What are the Border, Maritime, and Transportation Security Responsibilities and Functions?"

# --- Auto Tuning Config ---
AUTO_TUNING_CONFIG = {
    "domain": "document analysis",  # You can customize this based on your content
    "selection_method": "random",   # Options: random, top, all, auto
    "limit": 15,                   # Number of text units to use for prompt generation
    "language": "English",
    "max_tokens": 2048,
    "chunk_size": 256,
    "min_examples_required": 3,
    "discover_entity_types": False,  # Set to True if you want automatic entity discovery
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
    with open(html_path, "r", encoding="utf-8") as f:
        html_str = f.read()
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
# Check if chunks already exist
chunk_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("chunk_") and f.endswith(".txt")]
if chunk_files:
    print(f"[INFO] Found {len(chunk_files)} existing chunk files, skipping chunk generation")
else:
    print("[INFO] Generating chunks from HTML file...")
    chunks = parse_html_to_chunks("PRELIMusc06_sample.htm")
    for i, chunk in enumerate(chunks):
        path = os.path.join(INPUT_DIR, f"chunk_{i:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(chunk)
    print(f"[INFO] Wrote {len(chunks)} chunk files into {INPUT_DIR}")

# --- Initialize workspace (if not already initialized) ---
if not (os.path.exists(os.path.join(WORKSPACE, ".env")) and os.path.exists(os.path.join(WORKSPACE, "settings.yaml"))):
    print("[INFO] Initializing GraphRAG workspace...")
    subprocess.run([sys.executable, "-m", "graphrag", "init", "--root", WORKSPACE], check=True)
else:
    print("[INFO] GraphRAG workspace already initialized.")

# --- Run auto prompt tuning ---
prompts_dir = os.path.join(WORKSPACE, AUTO_TUNING_CONFIG["output"])
prompt_files = ["entity_extraction.txt", "summarize_descriptions.txt", "community_report.txt"]

# Check if auto-tuning was already completed by checking if prompts directory exists and has files
if os.path.exists(prompts_dir) and os.listdir(prompts_dir):
    prompt_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
    print(f"[INFO] Found existing prompts directory with {len(prompt_files)} prompt files, skipping auto tuning")
    print(f"[INFO] Auto-tuning was previously completed successfully")
else:
    print("[INFO] Running auto prompt tuning for domain-adapted prompts...")
    tuning_success = run_auto_prompt_tuning(WORKSPACE)
    
    if tuning_success:
        print("[INFO] Auto-tuning completed successfully - GraphRAG will use the tuned prompts automatically")
    else:
        print("[WARNING] Auto prompt tuning failed, proceeding with default prompts")
"""

# --- Index the input file ---
# Check if indexing is already complete by looking for output files
output_dir = os.path.join(WORKSPACE, "output")
indexing_complete = False

if os.path.exists(output_dir):
    # Check for key output files that indicate indexing is complete
    key_files = ["entities.parquet", "relationships.parquet", "communities.parquet"]
    existing_files = [f for f in key_files if os.path.exists(os.path.join(output_dir, f))]
    if len(existing_files) >= 2:  # At least 2 of the 3 key files exist
        print(f"[INFO] Found existing indexing output ({len(existing_files)}/3 key files), skipping indexing")
        print(f"[INFO] Existing files: {', '.join(existing_files)}")
        indexing_complete = True

if not indexing_complete:
    print("[INFO] Indexing input file with GraphRAG...")
    # Set environment variable to disable numba caching to avoid hyppo library issues
    env = os.environ.copy()
    env["NUMBA_DISABLE_JIT"] = "1"
    subprocess.run([sys.executable, "-m", "graphrag", "index", "--root", WORKSPACE], check=True, env=env)
else:
    print("[INFO] Using existing indexed data")

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