#!/bin/bash
# Inlines the case data into the console template. Artifacts must be self-contained,
# so the JSON cannot be fetched at runtime.
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
tpl = open("console_template.html").read()
data = open("dist/console_data.json").read()
assert "__DATA__" in tpl, "template placeholder missing"
assert "</script" not in data, "payload would close the script tag early"
out = tpl.replace("__DATA__", data)
open("dist/console.html", "w").write(out)
print(f"dist/console.html  {len(out)/1e6:.2f} MB")
PY
