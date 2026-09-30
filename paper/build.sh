set -euo pipefail

cd "$(dirname "$0")"

if [[ "${1:-}" == "--watch" ]]; then
    latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error -outdir=. main.tex
else
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=. main.tex
fi