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

show_compare_results() {
    python3 - <<'PYEOF'
import json, os, re
cj = 'outputs/eval/comparison.json'
md = 'outputs/eval/details/Teacher_R1-Distill-Qwen-7B.md'
if not os.path.isfile(cj):
    raise SystemExit
G, C, B, N = '\033[0;32m', '\033[0;36m', '\033[1m', '\033[0m'
print(f"  {B}Latest compare{N} — R1 distillation, full 228 problems, all test cases")
teacher_live = None
for e in json.load(open(cj)):
    if e['name'].startswith('Teacher'):
        if 'R1' in e['name']:
            teacher_live = e
        continue
    label = 'Student distilled (R1 teacher)' if 'distilled' in e['name'] else 'Student original'
    print(f"    {label:<31} {G}{e['problems_solved']:>3}/{e['num_problems']}{N} solved   pass@1 {e['pass_at_1']*100:4.1f}%   pass@5 {e['pass_at_k']*100:4.1f}%   test cases {e['test_pass_rate']*100:4.1f}%")
if teacher_live:
    e = teacher_live
    print(f"    {'Teacher R1 Distill Qwen 7B':<31} {G}{e['problems_solved']:>3}/{e['num_problems']}{N} solved   pass@1 {e['pass_at_1']*100:4.1f}%   pass@5 {e['pass_at_k']*100:4.1f}%   test cases {e['test_pass_rate']*100:4.1f}%")
elif os.path.isfile(md):
    n = [int(x) for x in re.findall(r'## [✓✗] Problem \d+.*?—\s+(\d)/5 samples passed', open(md).read())]
    if n:
        solved = sum(1 for x in n if x)
        print(f"    {'Teacher R1 Distill Qwen 7B':<31} {G}{solved:>3}/{len(n)}{N} solved   pass@1 {100*sum(n)/(5*len(n)):4.1f}%   pass@5 {100*solved/len(n):4.1f}%   {C}archived Jul 6 eval, old harness{N}")
PYEOF
}

show_menu() {
    local teacher_path student_model cache_dir
    teacher_path=$(grep 'local_model_path' config/config.yaml 2>/dev/null | awk '{print $2}')
    student_model=$(grep 'model_name' config/config.yaml 2>/dev/null | awk '{print $2}')
    cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
    header
    echo -e "  Teacher: ${GREEN}R1-Distill-Qwen-7B${NC}  ${CYAN}(local weights: ${teacher_path})${NC}"
    echo -e "  Student: ${GREEN}${student_model}${NC}"
    echo ""
    show_cache_status "$cache_dir"
    echo ""
    show_compare_results
    echo ""
    echo -e "  ${BOLD}Training${NC}"
    echo "  1) Run training              (offline on R1 cache, then train, then compare)"
    echo "  2) Compare original | teacher | distilled + graph"
    echo "  8) On-policy GKD round       (student generates, R1 scores, pure-KD fine-tune, compare)"
    echo ""
    echo -e "  ${BOLD}Cache${NC}"
    echo "  3) Reset teacher cache (delete all OR failed-only)"
    echo "  4) Re-test failed cache        (re-run harness on failed entries, CPU-only)"
    echo "  5) Diagnose cache              (prompt-match + failure causes, CPU-only)"
    echo "  6) Build FULL-DIST cache       (R1 rescores its own traces, full top-20 logits, GPU)"
    echo "  7) Free disk                   (delete redundant checkpoint backups)"
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

ensure_gpu_free() {
    local running
    running=$(sudo docker ps --filter name=knowledge-distillation-quantization --format '{{.Names}}  ({{.Status}})')
    [ -z "$running" ] && return 0
    echo -e "  ${YELLOW}GPU is held by a running pipeline container:${NC}"
    echo "    $running"
    echo -n "  Stop it and continue? (yes/n): "
    read -r gpu_ans
    if [ "$gpu_ans" = "yes" ]; then
        sudo docker ps -q --filter name=knowledge-distillation-quantization | xargs -r sudo docker stop >/dev/null
        echo -e "  ${GREEN}✓ GPU cleared.${NC}"
        return 0
    fi
    echo -e "  Keeping the running job — returning to menu."
    return 1
}

CFG=config/config.yaml

# Pick the student config (instruct vs general base). Sets: CFG
choose_config() {
    echo -e "  ${BOLD}Student config${NC}"
    echo -e "     ${GREEN}1${NC}  instruct     (config/config.yaml — Qwen2.5-Coder-1.5B-Instruct)"
    echo -e "     2  general base (config/config_general.yaml — Qwen2.5-1.5B)"
    echo -n "  config [default: 1]: "
    read -r cfg_choice
    if [ "$cfg_choice" = "2" ]; then
        CFG=config/config_general.yaml
    else
        CFG=config/config.yaml
    fi
    echo -e "  → using ${GREEN}${CFG}${NC}"
    echo ""
}

# Fixed compare-eval parameters — no prompts, always full 3-way paper-grade run.
# Sets: CMP_SKIP_FLAG, CMP_NP, CMP_DIFFICULTY, COMPARE_AFTER
set_fixed_compare_params() {
    local eval_mnt
    eval_mnt=$(grep 'max_new_tokens' "$CFG" 2>/dev/null | awk '{print $2}')
    CMP_SKIP_FLAG=""
    CMP_NP=228
    CMP_DIFFICULTY="all"
    COMPARE_AFTER=1
    echo -n "  Compare against the teacher too? (yes/no) [default: yes]: "
    read -r cmp_teacher
    local models_line="original + teacher + distilled  (3-way)"
    if [ "$cmp_teacher" = "no" ] || [ "$cmp_teacher" = "n" ]; then
        CMP_SKIP_FLAG="--skip-teacher"
        models_line="original + distilled  (teacher skipped)"
    fi
    echo -e "  ${BOLD}Fixed compare config${NC}:"
    echo "    dataset      : LeetCode test split"
    echo "    models       : ${models_line}"
    echo "    num_problems : 228  (full test split: 48 easy / 101 medium / 79 hard)"
    echo "    difficulty   : all  (per-difficulty breakdown reported)"
    echo "    samples/prob : 5   temperature 0.6   top_p 0.95"
    echo "    max_tokens   : ${eval_mnt}   seed 1234"
    echo "    chunk_size   : 24 student / 8 teacher"
    echo "    reuse        : cached rows reused; distilled always re-runs"
}

compare_cache_status() {
    [ -d outputs/eval/intermediate ] || { echo -e "  ${YELLOW}No cached evals yet — all models will run.${NC}"; echo ""; return 0; }
    local student_model
    student_model=$(grep 'model_name' "$CFG" 2>/dev/null | awk '{print $2}' | head -1)
    echo -e "  ${BOLD}Cached eval status${NC} for num_problems=${CMP_NP}, difficulty=${CMP_DIFFICULTY}:"
    python3 - "$CMP_NP" "$CMP_DIFFICULTY" "$student_model" "$CMP_SKIP_FLAG" <<'PY'
import json, os, sys
np = int(sys.argv[1]); diff = sys.argv[2]
base = sys.argv[3].split("/")[-1]
skip_teacher = len(sys.argv) > 4 and sys.argv[4] == "--skip-teacher"

def safe(label):
    return label.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

checks = [f"Student original ({base})"]
if not skip_teacher:
    checks.append("Teacher (R1-Distill-Qwen-7B)")
d = "outputs/eval/intermediate"
for label in checks:
    p = os.path.join(d, safe(label) + ".json")
    reused = False
    if os.path.exists(p):
        try:
            c = json.load(open(p))
            reused = (c.get("num_problems") == np and c.get("difficulty") == diff
                      and c.get("num_samples") == 5 and c.get("k") == 5
                      and c.get("temperature") == 0.6 and c.get("top_p") == 0.95)
        except Exception:
            reused = False
    print(f"    {label:<50} {'REUSED (cached, skipped)' if reused else 'will RUN'}")
print(f"    {f'Student distilled ({base})':<50} always RUN (new checkpoint)")
PY
    echo ""
}

# Run compare-eval using already-collected CMP_* params (NO prompts).
run_compare_with_params() {
    compare_cache_status
    run_cmd "sudo docker compose run --rm compare_eval python scripts/compare_eval.py --config $CFG --num-problems $CMP_NP --difficulty $CMP_DIFFICULTY --num-samples 5 --temperature 0.6 --top-p 0.95 $CMP_SKIP_FLAG"
    echo -e "  Graph saved to: ${GREEN}outputs/eval/comparison.png${NC}"
}

run_compare_eval() {
    header
    echo -e "  ${BOLD}Compare original | teacher | distilled on LeetCode test split${NC}"
    echo ""
    choose_config
    ensure_gpu_free || return

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
    python3 - "$dir" <<'PYEOF'
import glob, json, os, re, sys
from collections import Counter

cache_dir = sys.argv[1]
files = glob.glob(os.path.join(cache_dir, '*.json'))
pat_p = re.compile(rb'"test_passed":\s*(\d+)')
pat_t = re.compile(rb'"test_total":\s*(\d+)')

prob_file = None
for candidate in glob.glob("cache/problems_*.json"):
    prob_file = candidate
    break

difficulties = {}
if prob_file and os.path.exists(prob_file):
    try:
        probs = json.load(open(prob_file))
        for i, p in enumerate(probs):
            difficulties[i] = p.get("difficulty", "") or "?"
    except Exception:
        pass

G = '\033[0;32m'
R = '\033[0;31m'
Y = '\033[1;33m'
B = '\033[1m'
C = '\033[0;36m'
N = '\033[0m'

diff_pass = Counter()
diff_fail = Counter()
diff_fail_causes = {}

causes_by_idx = {}
try:
    with open('outputs/eval/diagnostics/failure_causes.json') as fh:
        causes_by_idx = json.load(fh).get('per_problem', {})
except Exception:
    pass

for f in files:
    idx = int(os.path.basename(f).replace('.json', ''))
    diff = difficulties.get(idx, "?")
    try:
        with open(f, 'rb') as fh:
            sz = os.fstat(fh.fileno()).st_size
            fh.seek(max(0, sz - 400))
            tail = fh.read()
        mp, mt = pat_p.search(tail), pat_t.search(tail)
        if mp and mt and int(mt.group(1)) > 0:
            if int(mp.group(1)) == int(mt.group(1)):
                diff_pass[diff] += 1
            else:
                diff_fail[diff] += 1
                fc = re.search(rb'"fail_cause":\s*"([^"]+)"', tail)
                if fc:
                    cause = fc.group(1).decode()
                else:
                    info = causes_by_idx.get(str(idx))
                    cause = info.get("dominant", "undiagnosed") if info else "undiagnosed (run option 5)"
                diff_fail_causes.setdefault(diff, Counter())[cause] += 1
        else:
            diff_fail[diff] += 1
    except Exception:
        diff_fail[diff] += 1

total_pass = sum(diff_pass.values())
total_fail = sum(diff_fail.values())
total = total_pass + total_fail
pct = total_pass * 100 // total if total > 0 else 0
print(f"    Passing (usable)     : {G}{total_pass}/{total}{N} ({pct}%)")
print()
print(f"    {B}{'Difficulty':<10} {'Pass':>6} {'Fail':>6} {'Total':>6} {'Rate':>6}  Fail causes{N}")
print(f"    {'─'*64}")
order = {"Easy": 0, "Medium": 1, "Hard": 2}
all_diffs = sorted(set(list(diff_pass) + list(diff_fail)), key=lambda d: order.get(d, 3))
for d in all_diffs:
    p = diff_pass.get(d, 0)
    fl = diff_fail.get(d, 0)
    t = p + fl
    r = f"{p*100//t}%" if t > 0 else "-"
    causes = diff_fail_causes.get(d, {})
    cause_str = ", ".join(f"{c}={n}" for c, n in sorted(causes.items(), key=lambda x: -x[1])) if causes else "-"
    color = G if p*100//max(t,1) >= 80 else (Y if p*100//max(t,1) >= 50 else R)
    print(f"    {d:<10} {p:>6} {fl:>6} {t:>6} {color}{r:>6}{N}  {cause_str}")
PYEOF
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
            choose_config
            cfg_epochs=$(grep 'num_epochs' "$CFG" | awk '{print $2}')
            cfg_alpha=$(grep 'alpha:' "$CFG" | awk '{print $2}' | head -1)
            cfg_lr=$(grep 'learning_rate' "$CFG" | awk '{print $2}')
            cfg_seed=$(grep 'seed:' "$CFG" | awk '{print $2}' | tail -1)
            cfg_temp=$(grep 'distill_temperature' "$CFG" | awk '{print $2}')
            cfg_alen=$(grep 'max_length' "$CFG" | awk '{print $2}' | head -1)
            cache_dir=$(grep 'teacher_cache_dir' "$CFG" | awk '{print $2}')
            echo -e "  ${BOLD}Training${NC} (offline train on R1 cache, then full 3-way compare)"
            echo ""
            echo -e "  ${BOLD}Fixed config${NC} (edit ${CFG} to change):"
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
            echo -e "     1  build/refresh missing entries first with the R1 teacher  (slow — regenerates every missing/failed problem)"
            echo -e "     ${GREEN}2${NC}  offline: train on the full-distribution R1 cache (best top-20 logprobs)"
            echo ""
            echo -n "  cache mode [default: 2]: "
            read -r cache_mode
            TRAIN_FLAGS="--config $CFG --max-samples $NS"
            if [ "$cache_mode" = "1" ]; then
                echo -e "  ${GREEN}→ Cache build/refresh with R1 teacher will run before training.${NC}"
            else
                if [ ! -d cache/teacher_logprobs_r1_full ]; then
                    echo -e "  ${RED}Full-distribution cache not found — run option 6 first.${NC}"
                    read -rp "Press Enter to return to menu..."
                    continue
                fi
                TRAIN_FLAGS="$TRAIN_FLAGS --offline --cache-dir cache/teacher_logprobs_r1_full"
                echo -e "  ${YELLOW}→ OFFLINE on full-distribution cache: R1's own full top-20 logits on its traces.${NC}"
            fi
            echo ""
            ensure_gpu_free || continue
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
            echo -e "  ${BOLD}Recover failed teacher cache${NC} (${cache_dir})"
            echo ""
            show_cache_status "$cache_dir"
            echo ""
            echo -e "  ${YELLOW}Note: re-test and retry change passing counts and therefore n_train; skip them to reproduce the control run exactly.${NC}"
            echo ""
            echo "  1) Re-test only (CPU, fast — no GPU needed)"
            echo "  2) Retry 3 candidates/problem with R1 teacher (GPU, no rescore)"
            echo "  3) Retry 5 candidates/problem with R1 teacher (GPU, no rescore)"
            echo "  q) Cancel"
            echo ""
            echo -n "  Select option: "
            read -r recover_choice
            case "$recover_choice" in
                1)
                    keep_sudo_alive
                    run_cmd "sudo docker compose run --rm compare_eval python scripts/rescore_tests.py --apply"
                    stop_sudo_alive
                    ;;
                2)
                    ensure_gpu_free || continue
                    keep_sudo_alive
                    run_cmd "sudo docker compose run --rm train python scripts/retry_failed_cache.py --attempts 3 --no-rescore"
                    stop_sudo_alive
                    ;;
                3)
                    ensure_gpu_free || continue
                    keep_sudo_alive
                    run_cmd "sudo docker compose run --rm train python scripts/retry_failed_cache.py --attempts 5 --no-rescore"
                    stop_sudo_alive
                    ;;
                *)
                    echo ""
                    echo -e "  Cancelled."
                    echo ""
                    read -rp "Press Enter to return to menu..."
                    ;;
            esac
            ;;
        5)
            header
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')

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
        6)
            header
            echo -e "  ${BOLD}Build FULL-DISTRIBUTION cache${NC} (pure R1 knowledge distillation)"
            echo ""
            echo -e "  The R1 teacher rescores its OWN traces via one teacher-forced pass at the"
            echo -e "  student's token positions (shared vocab table), recovering the FULL top-20"
            echo -e "  distribution everywhere (37.9% of current positions are one-hot from top-p truncation)."
            echo -e "  Source: cache/teacher_logprobs_coder15b  ->  cache/teacher_logprobs_r1_full"
            echo ""
            free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
            if [ "$free_gb" -lt 12 ]; then
                echo -e "  ${RED}Only ${free_gb}GB free — need ~12GB for the new cache. Run option 7 first.${NC}"
                echo ""
                read -rp "Press Enter to return to menu..."
                continue
            fi
            ensure_gpu_free || continue
            keep_sudo_alive
            run_cmd "sudo docker compose run --rm train python scripts/rescore_cache.py --src cache/teacher_logprobs_coder15b --dst cache/teacher_logprobs_r1_full --max-model-len 32768 --gpu-mem 0.80 --chunk-size 1"
            stop_sudo_alive
            ;;
        7)
            header
            echo -e "  ${BOLD}Free disk${NC} — redundant checkpoint backups:"
            echo ""
            for d in outputs/final outputs/final_baseline_bak; do
                [ -d "$d" ] && echo "    $(du -sh "$d" 2>/dev/null)"
            done
            echo ""
            echo -e "  ${CYAN}KEPT: outputs/final_r1_control (the 39/228 checkpoint, byte-identical to outputs/final)"
            echo -e "        outputs/final_last_29pct_bak (R1-era model, pending re-eval)"
            echo ""
            echo -n "  Delete the listed dirs? (yes/n): "
            read -r ans
            if [ "$ans" = "yes" ]; then
                sudo rm -rf outputs/final outputs/final_baseline_bak
                echo -e "  ${GREEN}✓ Deleted. Free now: $(df -h / | tail -1 | awk '{print $4}')${NC}"
            else
                echo "  Cancelled."
            fi
            echo ""
            read -rp "Press Enter to return to menu..."
            ;;
        8)
            header
            echo -e "  ${BOLD}On-policy GKD round${NC} — student generates, R1 scores its tokens, pure-KD fine-tune"
            echo ""
            echo -e "  ${YELLOW}Overwrites outputs/final (first offline model saved to outputs/final_offline_bak).${NC}"
            echo ""
            echo -n "  Problems to generate [blank = all train, or N for a quick test]: "
            read -r op_limit
            OP_LIMIT_FLAG=""
            [ -n "$op_limit" ] && OP_LIMIT_FLAG="--limit $op_limit"
            echo ""
            set_fixed_compare_params
            echo ""
            ensure_gpu_free || continue
            keep_sudo_alive
            [ -d outputs/final_offline_bak ] || sudo cp -r outputs/final outputs/final_offline_bak
            if ls cache/onpolicy_r1/*.json cache/onpolicy_r1_gen/*.json >/dev/null 2>&1; then
                echo -n "  Existing on-policy data found — (r)esume or (f)resh restart? [r]: "
                read -r op_mode
                [ "$op_mode" = "f" ] && sudo rm -rf cache/onpolicy_r1_gen cache/onpolicy_r1
            fi
            if run_cmd_noprompt "sudo docker compose run --rm train python scripts/onpolicy_generate.py $OP_LIMIT_FLAG" \
               && run_cmd_noprompt "sudo docker compose run --rm train python scripts/rescore_cache.py --src cache/onpolicy_r1_gen --dst cache/onpolicy_r1 --max-model-len 32768 --gpu-mem 0.80 --chunk-size 1"; then
                run_training_then_optionally_compare "sudo docker compose run --rm train python scripts/train.py --offline --onpolicy"
            else
                echo -e "  ${RED}On-policy generation/scoring failed — skipping training.${NC}"
                read -rp "Press Enter to return to menu..."
            fi
            stop_sudo_alive
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
