#!/usr/bin/env bash
# Downloads only what's needed for evaluation from Nutrition5k GCS bucket.
# Full dataset is 181.4 GB — this pulls splits + metadata (~10 MB) first,
# then optionally downloads test-split imagery on demand.
#
# Usage:
#   ./download_dataset.sh                    # splits + metadata only
#   ./download_dataset.sh --images           # also download test-split images (all cameras)
#   ./download_dataset.sh --images --cam A   # only camera A images (smallest subset)

set -euo pipefail

BUCKET="gs://nutrition5k_dataset/nutrition5k_dataset"
DATA_DIR="${NUTRITION5K_DIR:-./data/nutrition5k}"
CAMERA="${CAM:-}"
DOWNLOAD_IMAGES=false

for arg in "$@"; do
  case $arg in
    --images) DOWNLOAD_IMAGES=true ;;
    --cam)    shift; CAMERA="$1" ;;
  esac
done

mkdir -p "$DATA_DIR"

echo "==> Downloading splits..."
gsutil -m cp -r "$BUCKET/dish_ids/" "$DATA_DIR/"

echo "==> Downloading metadata CSVs..."
gsutil -m cp -r "$BUCKET/metadata/" "$DATA_DIR/"

echo "Done. Splits and metadata saved to $DATA_DIR"

if [ "$DOWNLOAD_IMAGES" = false ]; then
  echo ""
  echo "Image download skipped. Run with --images to pull test imagery."
  echo "Test dish IDs are in: $DATA_DIR/dish_ids/splits/rgb_test_ids.txt"
  exit 0
fi

TEST_IDS_FILE="$DATA_DIR/dish_ids/splits/rgb_test_ids.txt"
if [ ! -f "$TEST_IDS_FILE" ]; then
  echo "ERROR: $TEST_IDS_FILE not found. Run without --images first."
  exit 1
fi

echo "==> Downloading test-split imagery..."
echo "    ($(wc -l < "$TEST_IDS_FILE") dishes)"

while IFS= read -r dish_id; do
  [ -z "$dish_id" ] && continue
  if [ -n "$CAMERA" ]; then
    dst="$DATA_DIR/imagery/side_angles/$dish_id/camera_$CAMERA/"
    mkdir -p "$dst"
    gsutil -m cp "$BUCKET/imagery/side_angles/$dish_id/camera_$CAMERA/*" "$dst" 2>/dev/null || true
  else
    dst="$DATA_DIR/imagery/side_angles/$dish_id/"
    mkdir -p "$dst"
    gsutil -m cp -r "$BUCKET/imagery/side_angles/$dish_id/" "$dst" 2>/dev/null || true
  fi
done < "$TEST_IDS_FILE"

echo "Done. Images saved under $DATA_DIR/imagery/"
