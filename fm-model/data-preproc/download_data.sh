#!/bin/bash
set -euo pipefail

DATASET_JSON_FILE="dataset.json"
DATA_DIR="/home/plbenveniste/ms_fm_model/data"
CURRENT_DIR=$(pwd)

mkdir -p "$DATA_DIR"

if [[ ! -f "$DATASET_JSON_FILE" ]]; then
    echo "ERROR: File dataset.json not found in current directory"
    exit 1
fi

# Read each dataset entry as JSON object
datasets=$(jq -c '.datasets[] | {url: .source.url}' "$DATASET_JSON_FILE")

echo "Cloning datasets into: $DATA_DIR"
echo "-----------------------------------------------------"
cd "$DATA_DIR"
echo "$datasets" | while IFS= read -r ds; do
    url=$(echo "$ds" | jq -r '.url')

    if [[ -z "$url" || "$url" == "null" ]]; then
        echo "Skipping entry with no URL."
        continue
    fi

    echo "Cloning: $url"
    git clone "$url"
    echo ""
done

echo "-----------------------------------------------------"
echo "All datasets cloned successfully into $DATA_DIR"

FILES_JSON_FILE="files.json"

if [[ ! -f "$FILES_JSON_FILE" ]]; then    
    echo "Creating files JSON..."
    cd "$DATA_DIR"
    find ~+ -type f -name "*.nii.gz" \
        | grep -v "label" \
        | grep -v "MTS" \
        | grep -v "/derivatives/" \
        | sort > files.txt
    python "$CURRENT_DIR/create_files_json.py" --data_path "$DATA_DIR" --txt_file "$DATA_DIR/files.txt" --output_file "$DATA_DIR/$FILES_JSON_FILE"

fi

echo "Files JSON created at: $DATA_DIR/$FILES_JSON_FILE"

echo "Reading files from JSON and running git annex get..."
echo "------------------------------------------------------"

# Extract all image paths from TRAINING, VALIDATION, TESTING
ALL_FILES=$(jq -r '.TRAINING[].image, .VALIDATION[].image, .TESTING[].image' "$FILES_JSON_FILE")

echo "$ALL_FILES" | while IFS= read -r filepath; do
    if [[ -z "$filepath" ]]; then
        continue
    fi

    echo "Getting file: $filepath"

    # Extract dataset name (first directory in path)
    DATASET_NAME=$(echo "$filepath" | cut -d'/' -f1)

    DATASET_DIR="$DATA_DIR/$DATASET_NAME"

    if [[ ! -d "$DATASET_DIR/.git" ]]; then
        echo "❌ ERROR: $DATASET_DIR is not a git repository. Skipping."
        continue
    fi

    # Move into dataset repo
    cd "$DATASET_DIR"

    # Remove dataset name prefix for annex path
    REL_PATH="${filepath#${DATASET_NAME}/}"

    git annex get "$REL_PATH" || {
        echo "⚠️  Warning: failed to get $filepath"
    }

done
echo "------------------------------------------------------"