#!/bin/bash
# Renders the markdown write-ups to PDF (pandoc -> HTML fragment -> headless Chrome).
set -e
cd "$(dirname "$0")"
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
render () {  # $1 = source .md, $2 = output pdf name
  { echo '<!doctype html><meta charset="utf-8"><link rel="stylesheet" href="pdf.css">'
    pandoc "$1" -f gfm -t html5
  } > "dist/${2}.html"
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$PWD/dist/${2}.pdf" "file://$PWD/dist/${2}.html" 2>/dev/null
  rm -f "dist/${2}.html"
}
render SUMMARY.md OOS_Tag_Challenge_Summary
render METHOD.md  OOS_Tag_Challenge_Method
ls -lh dist/*.pdf | awk '{print $5, $9}'
