#!/usr/bin/env bash
# Render an SVG/HTML to PNG at an exact size with headless Chromium.
# usage: render.sh <file> <width> <height> <out.png> [bg=transparent]
set -euo pipefail
CH=${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}
f=$(realpath "$1"); w=$2; h=$3; out=$4; bg=${5:-transparent}
tmp=$(mktemp -d)
cat > "$tmp/r.html" <<HTML
<!doctype html><html><head><style>html,body{margin:0;background:$bg;overflow:hidden}img{display:block}</style></head>
<body><img src="file://$f" width="$w" height="$h"></body></html>
HTML
"$CH" --headless=new --no-sandbox --disable-gpu --hide-scrollbars --default-background-color=00000000 \
  --window-size="$w,$((h + 200))" --screenshot="$tmp/s.png" "file://$tmp/r.html" >/dev/null 2>&1
python3 -c "from PIL import Image; Image.open('$tmp/s.png').crop((0,0,$w,$h)).save('$out')"
rm -rf "$tmp"
