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
    echo "  1) Run training              (build/refresh teacher cache, then train)"
    echo "  2) Run training --offline    (skip cache build, use existing cache)"
    echo "  3) Run evaluation            (single checkpoint, val split)"
    echo "  4) Compare original | teacher | distilled + graph"
    echo ""
    echo -e "  ${BOLD}Cache${NC}"
    echo "  5) View cached teacher responses"
    echo "  6) Reset teacher cache (delete all cached responses)"
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

run_cmd() {
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
    echo ""
    read -rp "Press Enter to return to menu..."
}

view_cache() {
    local cache_dir
    cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
    cache_dir=${cache_dir:-cache/teacher_logprobs_reasoning}
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
            echo -e "  ${BOLD}Training${NC} (build/refresh teacher cache, then train)"
            echo ""
            echo -e "  ${BOLD}Fixed config${NC} (paper-grade defaults, edit config/config.yaml to change):"
            echo "    epochs              : ${cfg_epochs}     (Hinton 2015 / DistilBERT standard)"
            echo "    alpha (distill mix) : ${cfg_alpha}"
            echo "    distill temperature : ${cfg_temp}"
            echo "    learning rate       : ${cfg_lr}"
            echo "    max_length          : ${cfg_alen}"
            echo "    seed                : ${cfg_seed}"
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
            run_cmd "sudo docker compose run --rm train python train.py --max-samples $ns"
            ;;
        2)
            header
            REC_SAMPLES=1000
            cfg_epochs=$(grep 'num_epochs' config/config.yaml | awk '{print $2}')
            cfg_seed=$(grep 'seed:' config/config.yaml | awk '{print $2}')
            echo -e "  ${BOLD}Offline training${NC} (skip cache build, use existing cache)"
            echo ""
            echo -e "  Fixed config: epochs=${cfg_epochs}, seed=${cfg_seed} (edit config/config.yaml to change)"
            echo ""
            echo -n "  max_samples [default: ${REC_SAMPLES}]: "
            read -r ns
            [ -z "$ns" ] && ns=$REC_SAMPLES
            run_cmd "sudo docker compose run --rm train python train.py --offline --max-samples $ns"
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
                run_cmd "sudo docker compose run --rm evaluate python evaluate.py --checkpoint /workspace/$ckpt$vflag"
            fi
            ;;
        4)
            header
            echo -e "  ${BOLD}Compare original | teacher | distilled on LeetCode test split${NC}"
            echo ""
            echo -e "  ${BOLD}Fixed eval config${NC} (paper-grade defaults):"
            echo "    difficulty   : all"
            echo "    teacher eval : ON (never skipped)"
            echo "    samples/prob : 5  (pass@5 standard)"
            echo "    temperature  : 0.7"
            echo "    top_p        : 0.95"
            echo ""
            echo -e "  ${BOLD}num_problems${NC} — test problems to evaluate (LeetCode test split)"
            echo "     30    quick smoke test          (~10 min, noisy single-run)"
            echo "     100   ablation/intermediate     (~30 min, reasonable confidence)"
            echo -e "     ${GREEN}164${NC}   ${GREEN}HumanEval-comparable n${NC}  (~50 min, paper-grade)"
            echo "     full  entire test split        (~hours, most reliable)"
            echo ""
            echo -n "  num_problems [default: 100]: "
            read -r np
            [ -z "$np" ] && np=100
            run_cmd "sudo docker compose run --rm compare_eval python compare_eval.py --num-problems $np --difficulty all --num-samples 5 --temperature 0.7 --top-p 0.95"
            echo -e "  Graph saved to: ${GREEN}outputs/eval/comparison.png${NC}"
            ;;
        5)
            view_cache
            ;;
        6)
            header
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
            cache_dir=${cache_dir:-cache/teacher_logprobs_reasoning}
            count=$(find "$cache_dir" -name '*.json' 2>/dev/null | wc -l)
            echo -e "  Cache directory: ${YELLOW}${cache_dir}${NC}"
            echo -e "  Files found:     ${YELLOW}${count}${NC}"
            echo ""
            if [ "$count" -eq 0 ]; then
                echo -e "  ${GREEN}Cache is already empty.${NC}"
                echo ""
                read -rp "Press Enter to return to menu..."
            else
                echo -e "  ${RED}This will delete all $count cached response files.${NC}"
                echo -n "  Are you sure? (yes/n): "
                read -r ans
                if [ "$ans" = "yes" ]; then
                    sudo rm -f "${cache_dir}"/*.json
                    echo ""
                    echo -e "  ${GREEN}✓ Cache cleared.${NC}"
                else
                    echo ""
                    echo -e "  Cancelled."
                fi
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
