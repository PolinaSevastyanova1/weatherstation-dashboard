#!/bin/bash
# Runs hourly: pulls latest data from GitHub, then processes it.

BASE="/Users/polina.sevastyanova/Library/CloudStorage/OneDrive-UniversityofBergen/Desktop/Weather Station 1/processing_chain"
DATA_DIR="$BASE/data/L0"
SRC_DIR="$BASE/src"
DASH_DIR="$BASE/dashboard"
LOG="$BASE/scripts/hourly.log"

echo "" >> "$LOG"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
echo "-- Hourly Data Script Started --" | tee -a "$LOG"

# Pull latest raw data — reset to remote so local modifications never block the pull
if [ -d "$DATA_DIR/.git" ]; then
    cd "$DATA_DIR" && git fetch origin >> "$LOG" 2>&1 && git reset --hard origin/main >> "$LOG" 2>&1
else
    echo "WARN: data/L0 is not a git repo yet — skipping pull" | tee -a "$LOG"
fi

# Run processing script from src/ so relative paths resolve correctly
cd "$SRC_DIR" || { echo "ERROR: src/ not found" | tee -a "$LOG"; exit 1; }
PYTHON="/Users/polina.sevastyanova/Library/CloudStorage/OneDrive-UniversityofBergen/Desktop/.venv/bin/python3"
$PYTHON "$SRC_DIR/L0toL1.py" >> "$LOG" 2>&1
$PYTHON "$SRC_DIR/L1toL2.py" >> "$LOG" 2>&1

cd "$DASH_DIR" || { echo "ERROR: dashboard folder not found" | tee -a "$LOG"; exit 1; }

# Remove stale lock if a previous run crashed mid-git
rm -f "$DASH_DIR/.git/index.lock"

# Only commit files that change each run; index.html and static assets are committed once manually
git add data.json windrose.png temp_pressure_plot.png >> "$LOG" 2>&1
git diff --cached --quiet || git commit -m "data $(date '+%Y-%m-%d %H:%M')" >> "$LOG" 2>&1
git push origin main >> "$LOG" 2>&1


echo "-- Hourly Data Script Finished --" | tee -a "$LOG"
