#!/bin/bash
SCRIPT="$(dirname "$(realpath "$0")")/eval_array_W8A16_INT8KV.sh"

# humaneval(0), mbpp(1), winogrande(5), gpqa(6), gsm8k(7) — 12.5 min
sbatch --array=0,1,5,6,7 --time=00:31:15 "$SCRIPT"

# hellaswag(4) — 1 hour
sbatch --array=4 --time=02:30:00 "$SCRIPT"

# mmlu(2) — 2 hours, 64GB RAM
sbatch --array=2 --time=05:00:00 --mem=64G "$SCRIPT"

# # mmlu_pro(3) — 4 hours, 4 cores, 48GB RAM
# sbatch --array=3 --time=04:00:00 --mem=48G -c4 "$SCRIPT"

# wikitext(8) — 13 min
sbatch --array=8 --time=00:32:30 "$SCRIPT"
