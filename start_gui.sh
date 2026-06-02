#!/bin/bash
set -e
cd "$(dirname "$0")"

# Bird record export (.xls) needs xlutils in the same Python as below
if ! python -c "import xlutils" 2>/dev/null; then
    echo "[Birdy] Installing xlrd xlwt xlutils for record export..."
    python -m pip install xlrd xlwt xlutils pypinyin
fi

cd src
echo "Starting Birdy Skill GUI..."
python birdy_gui.py
