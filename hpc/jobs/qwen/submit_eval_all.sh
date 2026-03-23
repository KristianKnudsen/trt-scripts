#!/bin/bash
# Submit eval jobs for all quants
DIR="$(dirname "$(realpath "$0")")"

bash "$DIR/submit_eval_W16A16.sh"
bash "$DIR/submit_eval_W8A16.sh"
bash "$DIR/submit_eval_W8A8_SQ.sh"
bash "$DIR/submit_eval_W4A16.sh"
bash "$DIR/submit_eval_W4A16_AWQ.sh"
