#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

header() {
    clear
    echo -e "${BOLD}${CYAN}"
    echo "╔══════════════════════════════════════════════╗"
    echo "║    QEAD Knowledge Distillation Pipeline      ║"
    echo "╚══════════════════════════════════════════════╝"
    echo -e "${NC}"
}

show_menu() {
    local teacher_path student_model
    teacher_path=$(grep 'local_model_path' config/config.yaml 2>/dev/null | awk '{print $2}')
    student_model=$(grep 'model_name' config/config.yaml 2>/dev/null | awk '{print $2}')
    header
    echo -e "  Teacher: ${GREEN}${teacher_path}${NC}  ${CYAN}(local R1-Distill-Qwen-7B, bf16)${NC}"
    echo -e "  Student: ${GREEN}${student_model}${NC}"
    echo ""
    echo -e "  ${BOLD}Training${NC}"
    echo "  1) Run training              (build/refresh teacher cache, then train, then compare)"
    echo "  2) Run training --offline    (skip cache build, use existing cache)"
    echo "  3) Run evaluation            (single checkpoint, val split)"
    echo "  4) Compare original | teacher | distilled + graph"
    echo ""
    echo -e "  ${BOLD}Cache${NC}"
    echo "  5) View cached teacher responses"
    echo "  6) Reset teacher cache (delete all OR failed-only)"
    echo ""
    echo -e "  ${BOLD}Docker${NC}"
    echo "  7) Build Docker image"
    echo "  8) Open shell inside container"
    echo ""
    echo -e "  ${BOLD}GPU Setup${NC}"
    echo "  9) Install nvidia-container-toolkit (requires sudo)"
    echo ""
    echo "  q) Quit"
    echo ""
    echo -n "  Select option: "
}

run_cmd_noprompt() {
    echo -e "${YELLOW}▶ $*${NC}"
    echo ""
    eval "$@"
    local code=$?
    echo ""
    if [ $code -eq 0 ]; then
        echo -e "${GREEN}✓ Done (exit 0)${NC}"
    else
        echo -e "${RED}✗ Failed (exit $code)${NC}"
    fi
    return $code
}

run_cmd() {
    run_cmd_noprompt "$@"
    local code=$?
    echo ""
    read -rp "Press Enter to return to menu..."
    return $code
}

# Ask for the sudo password ONCE up front and refresh it in the background, so a long
# multi-step pipeline never stops to re-ask mid-run (sudo's timestamp would otherwise
# expire between the train / retrain / compare steps).
keep_sudo_alive() {
    echo -e "  ${CYAN}Caching sudo credentials (asked once now, kept alive for the whole run)...${NC}"
    sudo -v || return 1
    ( while kill -0 "$$" 2>/dev/null; do sudo -n true 2>/dev/null; sleep 60; done ) &
    SUDO_KEEPALIVE_PID=$!
}

stop_sudo_alive() {
    [ -n "$SUDO_KEEPALIVE_PID" ] && kill "$SUDO_KEEPALIVE_PID" 2>/dev/null
    SUDO_KEEPALIVE_PID=""
}

# Collect compare-eval parameters into globals WITHOUT running anything.
# Sets: CMP_SKIP_FLAG, CMP_NP, CMP_DIFFICULTY
prompt_compare_params() {
    echo -e "  ${BOLD}Fixed eval config${NC} (paper-grade defaults):"
    echo "    samples/prob : 5  (pass@5 standard)"
    echo "    temperature  : 0.7   top_p : 0.95"
    echo "    chunk_size   : 10 student / 3 teacher   (scaled for 24576 budget)"
    echo "    max_tokens   : 24576  (paper-grade, target ~20% truncation)"
    echo "    seed         : 1234  (fixed → pass@5 reproducible run-to-run)"
    echo "    reuse        : teacher + original cached & reused; only distilled re-runs"
    echo ""
    echo -e "  ${BOLD}Include teacher in comparison?${NC}"
    echo -e "    ${GREEN}y${NC}  ALL 3 models (paper-grade comparison, ~3-4.5h total)"
    echo "    n  Student-only (skip teacher, ~1.5-2.5h, quick iteration)"
    echo ""
    echo -n "  Include teacher? (y/n) [default: y]: "
    read -r incl_teacher
    CMP_SKIP_FLAG=""
    if [ "$incl_teacher" = "n" ] || [ "$incl_teacher" = "N" ]; then
        CMP_SKIP_FLAG="--skip-teacher"
        echo -e "  ${YELLOW}→ Teacher SKIPPED. Will compare Student (original) vs Student (distilled) only.${NC}"
    else
        echo -e "  ${GREEN}→ Teacher INCLUDED. 3-way comparison.${NC}"
    fi
    echo ""
    echo -e "  ${BOLD}num_problems${NC} — test problems to evaluate (LeetCode test split)"
    echo "     30    quick smoke test          (~15 min, noisy single-run)"
    echo "     100   ablation/intermediate     (~45 min, reasonable confidence)"
    echo -e "     ${GREEN}164${NC}   ${GREEN}HumanEval-comparable n${NC}  (~75 min, paper-grade)"
    echo "     full  entire test split        (~hours, most reliable)"
    echo ""
    echo -n "  num_problems [default: 100]: "
    read -r CMP_NP
    [ -z "$CMP_NP" ] && CMP_NP=100
    echo ""
    echo -e "  ${BOLD}difficulty${NC} — which levels to include (applied equally to all models)"
    echo -e "     ${GREEN}1${NC}  all              (easy + medium + hard)"
    echo "     2  easy + medium   (exclude hard — report as 'Easy+Medium' only)"
    echo ""
    echo -n "  difficulty [default: 1]: "
    read -r diff_choice
    CMP_DIFFICULTY="all"
    if [ "$diff_choice" = "2" ]; then
        CMP_DIFFICULTY="easy,medium"
        echo -e "  ${YELLOW}→ Hard EXCLUDED. Remember to label results 'Easy+Medium' in any report.${NC}"
    fi
}

compare_cache_status() {
    [ -d outputs/eval/intermediate ] || { echo -e "  ${YELLOW}No cached evals yet — all models will run.${NC}"; echo ""; return 0; }
    echo -e "  ${BOLD}Cached eval status${NC} for num_problems=${CMP_NP}, difficulty=${CMP_DIFFICULTY}, pass@5:"
    python3 - "$CMP_NP" "$CMP_DIFFICULTY" <<'PY'
import json, os, sys
np = int(sys.argv[1]); diff = sys.argv[2]
checks = [("Student (original)", "Student_original.json"),
          ("Teacher (R1-Distill-Qwen-7B)", "Teacher_R1-Distill-Qwen-7B.json")]
d = "outputs/eval/intermediate"
for label, fn in checks:
    p = os.path.join(d, fn)
    reused = False
    if os.path.exists(p):
        try:
            c = json.load(open(p))
            reused = (c.get("num_problems") == np and c.get("difficulty") == diff
                      and c.get("num_samples") == 5 and c.get("k") == 5
                      and c.get("temperature") == 0.7 and c.get("top_p") == 0.95)
        except Exception:
            reused = False
    print(f"    {label:<32} {'REUSED (cached, skipped)' if reused else 'will RUN'}")
print(f"    {'Student (distilled)':<32} always RUN (new checkpoint)")
PY
    echo ""
}

# Run compare-eval using already-collected CMP_* params (NO prompts).
run_compare_with_params() {
    compare_cache_status
    run_cmd "sudo docker compose run --rm compare_eval python scripts/compare_eval.py --num-problems $CMP_NP --difficulty $CMP_DIFFICULTY --num-samples 5 --temperature 0.7 --top-p 0.95 $CMP_SKIP_FLAG"
    echo -e "  Graph saved to: ${GREEN}outputs/eval/comparison.png${NC}"
}

# Standalone compare (menu option 4): ask, then run.
run_compare_eval() {
    header
    echo -e "  ${BOLD}Compare original | teacher | distilled on LeetCode test split${NC}"
    echo ""
    prompt_compare_params
    echo ""
    run_compare_with_params
}

# Ask UP FRONT (before training runs) whether to auto-compare afterwards,
# and if so collect all compare params now so nothing is asked mid-run.
# Sets: COMPARE_AFTER (1/0) and CMP_* params.
prompt_compare_after() {
    echo -e "  ${BOLD}After training finishes, run compare eval automatically?${NC}"
    echo "    (everything is asked now — training then compare run unattended)"
    echo ""
    echo -n "  Compare after training? (y/n) [default: y]: "
    read -r run_cmp
    if [ "$run_cmp" = "n" ] || [ "$run_cmp" = "N" ]; then
        COMPARE_AFTER=0
        echo -e "  ${YELLOW}→ Will NOT compare after training.${NC}"
    else
        COMPARE_AFTER=1
        echo ""
        echo -e "  ${CYAN}── Compare options (collected now, run after training) ──${NC}"
        echo ""
        prompt_compare_params
    fi
}

# Run training, then auto-compare if it was requested up front (NO mid-run prompts).
run_training_then_optionally_compare() {
    local train_cmd="$1"
    run_cmd_noprompt "$train_cmd"
    local code=$?
    echo ""
    if [ $code -ne 0 ]; then
        echo -e "  ${RED}Training failed — skipping compare.${NC}"
        echo ""
        read -rp "Press Enter to return to menu..."
        return
    fi
    if [ "$COMPARE_AFTER" = "1" ]; then
        echo -e "  ${GREEN}✓ Training complete — starting pre-configured compare eval...${NC}"
        echo ""
        run_compare_with_params
        return
    fi
    echo -e "  ${GREEN}✓ Training complete. (compare skipped per your choice)${NC}"
    echo ""
    read -rp "Press Enter to return to menu..."
}

view_cache() {
    local cache_dir
    cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
    cache_dir=${cache_dir:-cache/teacher_logprobs_r1_7b}
    header
    local total
    total=$(find "$cache_dir" -name '*.json' 2>/dev/null | wc -l)
    if [ "$total" -eq 0 ]; then
        echo -e "${RED}No cached responses found in $cache_dir${NC}"
        echo ""
        read -rp "Press Enter to return to menu..."
        return
    fi
    echo -e "  Found ${GREEN}$total${NC} cached responses."
    echo -e "  Use ${BOLD}arrow keys / PgUp PgDn${NC} to scroll, ${BOLD}q${NC} to return.\n"
    python3 - "$cache_dir" <<'PYEOF' | less -R
import json, os, sys, textwrap, glob

cache_dir = sys.argv[1]
files = sorted(
    glob.glob(os.path.join(cache_dir, '*.json')),
    key=lambda f: int(os.path.basename(f).replace('.json', ''))
)

CYAN  = '\033[0;36m'
BOLD  = '\033[1m'
NC    = '\033[0m'

for f in files:
    num = os.path.basename(f).replace('.json', '')
    with open(f) as fh:
        d = json.load(fh)
    prompt = d.get('prompt', '').replace('\n', ' ').strip()
    text   = d.get('text', '').strip() or '(empty)'

    print(f'{BOLD}{CYAN}{"─" * 60}')
    print(f'  Entry #{num}  ({os.path.basename(f)}){NC}')
    print(f'{BOLD}{CYAN}{"─" * 60}{NC}')
    print(f'{BOLD}PROMPT:{NC}')
    for line in textwrap.wrap(prompt, width=78):
        print('  ' + line)
    print()
    print(f'{BOLD}RESPONSE:{NC}')
    for line in text.splitlines():
        print('  ' + line)
    print()
PYEOF
}

install_nvidia_toolkit() {
    header
    echo -e "${YELLOW}Installing nvidia-container-toolkit...${NC}"
    echo ""
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo ""
    echo -e "${GREEN}✓ nvidia-container-toolkit installed. Docker restarted.${NC}"
    echo ""
    read -rp "Press Enter to return to menu..."
}

show_cache_status() {
    local dir="$1"
    CACHE_TOTAL=0
    CACHE_MAX_IDX=-1
    echo -e "  ${BOLD}Teacher cache status${NC} (${dir})"
    if [ ! -d "$dir" ]; then
        echo -e "    ${RED}Cache dir not found${NC}"
        return 1
    fi
    CACHE_TOTAL=$(find "$dir" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    if [ "$CACHE_TOTAL" -eq 0 ]; then
        echo -e "    ${RED}No cached files${NC}"
        return 1
    fi
    CACHE_MAX_IDX=$(find "$dir" -maxdepth 1 -name '*.json' -printf '%f\n' 2>/dev/null | sed 's/\.json$//' | sort -n | tail -1)
    echo -e "    Total cached files   : ${GREEN}${CACHE_TOTAL}${NC}"
    echo -e "    Problem index range  : 0-${CACHE_MAX_IDX}  (max_samples ${GREEN}$((CACHE_MAX_IDX + 1))${NC} covers the full cache)"
    local stats
    stats=$(timeout 10 python3 -c "
import json, glob, random
files = glob.glob('$dir/*.json')
sample = random.sample(files, min(100, len(files)))
total = passed = 0
for f in sample:
    try:
        d = json.load(open(f))
        tp, tt = d.get('test_passed'), d.get('test_total')
        if tp is not None and tt is not None and tt > 0:
            total += 1
            if tp == tt: passed += 1
    except Exception: pass
print(f'{passed},{total}')
" 2>/dev/null)
    local s_pass s_total
    IFS=',' read -r s_pass s_total <<< "$stats"
    if [ -n "$s_total" ] && [ "$s_total" -gt 0 ]; then
        local pct=$((s_pass * 100 / s_total))
        echo -e "    Sample pass rate     : ${GREEN}${s_pass}/${s_total}${NC} (~${pct}% usable for training, random ${s_total}-file sample)"
    else
        echo -e "    Pass rate            : (skipped — scan timed out or files being written)"
    fi
    return 0
}

MENU_FILE="$(realpath "$0")"
MENU_MTIME="$(stat -c %Y "$MENU_FILE")"

while true; do
    current_mtime="$(stat -c %Y "$MENU_FILE")"
    if [ "$current_mtime" != "$MENU_MTIME" ]; then
        echo -e "\n${YELLOW}menu.sh changed — reloading...${NC}\n"
        exec "$MENU_FILE"
    fi
    show_menu
    read -r choice
    case $choice in
        1)
            header
            REC_SAMPLES=1000
            cfg_epochs=$(grep 'num_epochs' config/config.yaml | awk '{print $2}')
            cfg_alpha=$(grep 'alpha:' config/config.yaml | awk '{print $2}')
            cfg_lr=$(grep 'learning_rate' config/config.yaml | awk '{print $2}')
            cfg_seed=$(grep 'seed:' config/config.yaml | awk '{print $2}')
            cfg_temp=$(grep 'distill_temperature' config/config.yaml | awk '{print $2}')
            cfg_alen=$(grep 'max_length' config/config.yaml | awk '{print $2}')
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml | awk '{print $2}')
            echo -e "  ${BOLD}Training${NC} (build/refresh teacher cache, then train, then compare)"
            echo ""
            echo -e "  ${BOLD}Fixed config${NC} (paper-grade defaults, edit config/config.yaml to change):"
            echo "    epochs              : ${cfg_epochs}     (Hinton 2015 / DistilBERT standard)"
            echo "    alpha (distill mix) : ${cfg_alpha}"
            echo "    distill temperature : ${cfg_temp}"
            echo "    learning rate       : ${cfg_lr}"
            echo "    max_length          : ${cfg_alen}"
            echo "    seed                : ${cfg_seed}"
            echo ""
            if ! show_cache_status "$cache_dir"; then
                echo -e "    ${YELLOW}→ fresh generation will run${NC}"
            fi
            echo ""
            echo -e "  ${BOLD}max_samples${NC}  — problems to load (LeetCode train split has ~2,600)"
            echo "     200    quick smoke test          (~2 min teacher cache, low statistical power)"
            echo "     500    ablation runs             (~5 min teacher cache, ok for early experiments)"
            echo -e "     ${GREEN}1000${NC}   ${GREEN}recommended for paper${NC}   (~10 min teacher cache, solid n for pass@k)"
            echo "     2600   full dataset              (~25 min teacher cache, best — if you have time)"
            echo ""
            echo -n "  max_samples [default: ${REC_SAMPLES}]: "
            read -r ns
            [ -z "$ns" ] && ns=$REC_SAMPLES
            echo ""
            prompt_compare_after
            echo ""
            keep_sudo_alive
            run_training_then_optionally_compare "sudo docker compose run --rm train python scripts/train.py --max-samples $ns"
            stop_sudo_alive
            ;;
        2)
            header
            REC_SAMPLES=1000
            cfg_epochs=$(grep 'num_epochs' config/config.yaml | awk '{print $2}')
            cfg_seed=$(grep 'seed:' config/config.yaml | awk '{print $2}')
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml | awk '{print $2}')
            echo -e "  ${BOLD}Offline training${NC} (skip cache build, use existing cache)"
            echo ""
            echo -e "  Fixed config: epochs=${cfg_epochs}, seed=${cfg_seed} (edit config/config.yaml to change)"
            echo ""
            if show_cache_status "$cache_dir"; then
                REC_SAMPLES=$((CACHE_MAX_IDX + 1))
            else
                echo -e "    ${RED}→ offline training will fail without a cache${NC}"
            fi
            echo ""
            echo -n "  max_samples [default: ${REC_SAMPLES}]: "
            read -r ns
            [ -z "$ns" ] && ns=$REC_SAMPLES
            echo ""
            prompt_compare_after
            echo ""
            keep_sudo_alive
            run_training_then_optionally_compare "sudo docker compose run --rm train python scripts/train.py --offline --max-samples $ns"
            stop_sudo_alive
            ;;
        3)
            header
            ckpt="outputs/final"
            if [ ! -d "$ckpt" ]; then
                echo -e "${RED}No checkpoint at ${ckpt}. Run training first.${NC}"
                echo ""
                read -rp "Press Enter to return to menu..."
            else
                echo -e "Checkpoint: ${GREEN}${ckpt}${NC}"
                echo -n "Verbose output? (y/n) [default: n]: "
                read -r vb
                vflag=""
                [ "$vb" = "y" ] || [ "$vb" = "Y" ] && vflag=" --verbose"
                run_cmd "sudo docker compose run --rm evaluate python scripts/evaluate.py --checkpoint /workspace/$ckpt$vflag"
            fi
            ;;
        4)
            run_compare_eval
            ;;
        5)
            view_cache
            ;;
        6)
            header
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
            cache_dir=${cache_dir:-cache/teacher_logprobs_r1_7b}
            count=$(find "$cache_dir" -name '*.json' 2>/dev/null | wc -l)
            stats=$(python3 - "$cache_dir" <<'PYEOF'
import glob, os, re, sys
files = glob.glob(os.path.join(sys.argv[1], '*.json'))
pat_p = re.compile(rb'"test_passed":\s*(\d+)')
pat_t = re.compile(rb'"test_total":\s*(\d+)')
passed = failed = 0
for f in files:
    try:
        with open(f, 'rb') as fh:
            sz = os.fstat(fh.fileno()).st_size
            fh.seek(max(0, sz - 400))
            tail = fh.read()
        mp, mt = pat_p.search(tail), pat_t.search(tail)
        if mp and mt and int(mt.group(1)) and int(mp.group(1)) == int(mt.group(1)):
            passed += 1
        else:
            failed += 1
    except Exception:
        failed += 1
print(f'{passed} {failed}')
PYEOF
            )
            pass_count=$(echo "$stats" | awk '{print $1}')
            fail_count=$(echo "$stats" | awk '{print $2}')
            echo -e "  Cache directory: ${YELLOW}${cache_dir}${NC}"
            echo -e "  Total files:     ${YELLOW}${count}${NC}  (${GREEN}${pass_count} passing${NC}, ${RED}${fail_count} failed${NC})"
            echo ""
            if [ "$count" -eq 0 ]; then
                echo -e "  ${GREEN}Cache is already empty.${NC}"
                echo ""
                read -rp "Press Enter to return to menu..."
            else
                echo "  1) Delete ALL cached responses ($count files)"
                echo "  2) Delete FAILED only, keep passing ($fail_count files)"
                echo "  q) Cancel"
                echo ""
                echo -n "  Select option: "
                read -r reset_choice
                case "$reset_choice" in
                    1)
                        echo ""
                        echo -e "  ${RED}This will delete all $count cached response files.${NC}"
                        echo -n "  Are you sure? (yes/n): "
                        read -r ans
                        if [ "$ans" = "yes" ]; then
                            sudo rm -f "${cache_dir}"/*.json
                            echo ""
                            echo -e "  ${GREEN}✓ All cache cleared.${NC}"
                        else
                            echo ""
                            echo -e "  Cancelled."
                        fi
                        ;;
                    2)
                        echo ""
                        echo -e "  ${RED}This will delete $fail_count failed cached files, keep $pass_count passing.${NC}"
                        echo -n "  Are you sure? (yes/n): "
                        read -r ans
                        if [ "$ans" = "yes" ]; then
                            sudo python3 - "$cache_dir" <<'PYEOF'
import json, glob, os, sys
deleted = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    try:
        d = json.load(open(f))
        tp, tt = d.get('test_passed'), d.get('test_total')
        if not tt or tp is None or tp < tt:
            os.remove(f); deleted += 1
    except Exception:
        os.remove(f); deleted += 1
print(f'  deleted {deleted} failed files')
PYEOF
                            echo ""
                            echo -e "  ${GREEN}✓ Failed cache cleared.${NC}"
                        else
                            echo ""
                            echo -e "  Cancelled."
                        fi
                        ;;
                    *)
                        echo ""
                        echo -e "  Cancelled."
                        ;;
                esac
                echo ""
                read -rp "Press Enter to return to menu..."
            fi
            ;;
        7)
            header
            run_cmd "sudo docker compose build"
            ;;
        8)
            header
            echo -e "${YELLOW}▶ Opening shell inside train container...${NC}"
            echo ""
            sudo docker compose run --rm train bash
            echo ""
            read -rp "Press Enter to return to menu..."
            ;;
        9)
            install_nvidia_toolkit
            ;;
        q|Q)
            echo -e "${NC}Bye."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option.${NC}"
            sleep 1
            ;;
    esac
done
