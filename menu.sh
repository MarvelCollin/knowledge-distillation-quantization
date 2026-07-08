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
    echo "  2) Compare original | teacher | distilled + graph"
    echo ""
    echo -e "  ${BOLD}Cache${NC}"
    echo "  3) Reset teacher cache (delete all OR failed-only)"
    echo "  4) Rescore failed cache        (recover harness-fixed entries, CPU-only)"
    echo "  5) Diagnose cache              (prompt-match + failure causes, CPU-only)"
    echo ""
    echo "  q) Quit"
    echo ""
    echo -n "  Select option: "
}

run_cmd_noprompt() {
    echo -e "${YELLOW}▶ $*${NC}"
    echo ""
    mkdir -p logs
    local log="logs/$(date +%Y%m%d_%H%M%S).log"
    echo -e "${CYAN}Logging to $log${NC}"
    echo ""
    { echo "\$ $*"; echo ""; eval "$@"; } 2>&1 | tee "$log"
    local code=${PIPESTATUS[0]}
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

# Fixed compare-eval parameters — no prompts, always full 3-way paper-grade run.
# Sets: CMP_SKIP_FLAG, CMP_NP, CMP_DIFFICULTY, COMPARE_AFTER
set_fixed_compare_params() {
    local eval_mnt
    eval_mnt=$(grep 'max_new_tokens' config/config.yaml 2>/dev/null | awk '{print $2}')
    CMP_SKIP_FLAG=""
    CMP_NP=228
    CMP_DIFFICULTY="all"
    COMPARE_AFTER=1
    echo -e "  ${BOLD}Fixed compare config${NC} (auto, no prompts):"
    echo "    dataset      : LeetCode test split"
    echo "    models       : original + teacher + distilled  (3-way)"
    echo "    num_problems : 228  (full test split: 48 easy / 101 medium / 79 hard)"
    echo "    difficulty   : all  (per-difficulty breakdown reported)"
    echo "    samples/prob : 5   temperature 0.7   top_p 0.95"
    echo "    max_tokens   : ${eval_mnt}   seed 1234"
    echo "    chunk_size   : 24 student / 8 teacher"
    echo "    reuse        : teacher + original cached & reused; only distilled re-runs"
}

compare_cache_status() {
    [ -d outputs/eval/intermediate ] || { echo -e "  ${YELLOW}No cached evals yet — all models will run.${NC}"; echo ""; return 0; }
    echo -e "  ${BOLD}Cached eval status${NC} for num_problems=${CMP_NP}, difficulty=${CMP_DIFFICULTY}:"
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

run_compare_eval() {
    header
    echo -e "  ${BOLD}Compare original | teacher | distilled on LeetCode test split${NC}"
    echo ""
    set_fixed_compare_params
    echo ""
    run_compare_with_params
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
    cache_dir=${cache_dir:-cache/teacher_logprobs_coder15b}
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
    stats=$(python3 - "$dir" <<'PYEOF'
import glob, os, re, sys
files = glob.glob(os.path.join(sys.argv[1], '*.json'))
pat_p = re.compile(rb'"test_passed":\s*(\d+)')
pat_t = re.compile(rb'"test_total":\s*(\d+)')
passed = tested = 0
for f in files:
    try:
        with open(f, 'rb') as fh:
            sz = os.fstat(fh.fileno()).st_size
            fh.seek(max(0, sz - 400))
            tail = fh.read()
        mp, mt = pat_p.search(tail), pat_t.search(tail)
        if mp and mt and int(mt.group(1)) > 0:
            tested += 1
            if int(mp.group(1)) == int(mt.group(1)):
                passed += 1
    except Exception:
        pass
print(f'{passed},{tested}')
PYEOF
)
    local s_pass s_total
    IFS=',' read -r s_pass s_total <<< "$stats"
    if [ -n "$s_total" ] && [ "$s_total" -gt 0 ]; then
        local pct=$((s_pass * 100 / s_total))
        echo -e "    Passing (usable)     : ${GREEN}${s_pass}/${s_total}${NC} (${pct}% of tested cache, exact count)"
    else
        echo -e "    Pass rate            : (no tested entries yet)"
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
            NS=2600
            cfg_epochs=$(grep 'num_epochs' config/config.yaml | awk '{print $2}')
            cfg_alpha=$(grep 'alpha:' config/config.yaml | awk '{print $2}')
            cfg_lr=$(grep 'learning_rate' config/config.yaml | awk '{print $2}')
            cfg_seed=$(grep 'seed:' config/config.yaml | awk '{print $2}')
            cfg_temp=$(grep 'distill_temperature' config/config.yaml | awk '{print $2}')
            cfg_alen=$(grep 'max_length' config/config.yaml | awk '{print $2}')
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml | awk '{print $2}')
            echo -e "  ${BOLD}Training${NC} (build/refresh teacher cache, then train, then full 3-way compare)"
            echo ""
            echo -e "  ${BOLD}Fixed config${NC} (edit config/config.yaml to change):"
            echo "    max_samples         : ${NS}  (full LeetCode train split)"
            echo "    epochs              : ${cfg_epochs}"
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
            set_fixed_compare_params
            echo ""
            echo -e "  ${BOLD}Teacher cache mode${NC} (the only question)"
            echo -e "     ${GREEN}1${NC}  build/refresh missing entries first  (slow — regenerates every missing/failed problem)"
            echo "     2  offline: train NOW on the current passing cache only (skips generation entirely)"
            echo ""
            echo -n "  cache mode [default: 1]: "
            read -r cache_mode
            TRAIN_FLAGS="--max-samples $NS"
            if [ "$cache_mode" = "2" ]; then
                TRAIN_FLAGS="$TRAIN_FLAGS --offline"
                echo -e "  ${YELLOW}→ OFFLINE: no teacher generation; training uses existing passing cache as-is.${NC}"
            else
                echo -e "  ${GREEN}→ Cache build/refresh will run before training.${NC}"
            fi
            echo ""
            keep_sudo_alive
            run_training_then_optionally_compare "sudo docker compose run --rm train python scripts/train.py $TRAIN_FLAGS"
            stop_sudo_alive
            ;;
        2)
            run_compare_eval
            ;;
        3)
            header
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
            cache_dir=${cache_dir:-cache/teacher_logprobs_coder15b}
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
import glob, os, re, sys
pat_p = re.compile(rb'"test_passed":\s*(\d+)')
pat_t = re.compile(rb'"test_total":\s*(\d+)')
deleted = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    try:
        with open(f, 'rb') as fh:
            sz = os.fstat(fh.fileno()).st_size
            fh.seek(max(0, sz - 400))
            tail = fh.read()
        mp, mt = pat_p.search(tail), pat_t.search(tail)
        keep = bool(mp and mt) and int(mt.group(1)) > 0 and int(mp.group(1)) == int(mt.group(1))
    except Exception:
        keep = False
    if not keep:
        try:
            os.remove(f); deleted += 1
        except OSError:
            pass
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
        4)
            header
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
            cache_dir=${cache_dir:-cache/teacher_logprobs_coder15b}
            echo -e "  ${BOLD}Rescore failed teacher cache${NC} (${cache_dir})"
            echo ""
            echo -e "  Re-runs the unit tests on FAILED cached trajectories with the current"
            echo -e "  harness and marks any that now PASS as usable for training."
            echo -e "  ${CYAN}Pure knowledge distillation${NC}: teacher tokens/logprobs are never changed —"
            echo -e "  only wrongly-failed correct trajectories get un-discarded."
            echo -e "  ${GREEN}CPU-only${NC} (no GPU), safe to run alongside a GPU job."
            echo ""
            echo -n "  Rescore and update cache now? (y/n): "
            read -r ans
            if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
                run_cmd "sudo docker compose run --rm compare_eval python scripts/rescore_tests.py --apply"
            else
                echo ""
                echo -e "  Cancelled."
                echo ""
                read -rp "Press Enter to return to menu..."
            fi
            ;;
        5)
            header
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
            cache_dir=${cache_dir:-cache/teacher_logprobs_coder15b}
            echo -e "  ${BOLD}Diagnose teacher cache${NC} (${cache_dir})"
            echo ""
            echo -e "  Two CPU-only checks (no GPU, safe alongside a GPU job):"
            echo -e "    ${CYAN}prompt-match${NC} : how many cached prompts still match the current builder"
            echo -e "                   (stale prompts = passing entries pointlessly re-generated)"
            echo -e "    ${CYAN}failure-cause${NC}: why the failing trajectories fail"
            echo -e "                   (syntax_error / wrong_answer / timeout / missing_function)"
            echo ""
            run_cmd_noprompt "sudo docker compose run --rm compare_eval python scripts/check_cache_prompts.py"
            echo ""
            run_cmd "sudo docker compose run --rm compare_eval python scripts/analyze_failures.py"
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
