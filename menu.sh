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
    local cur_model
    cur_model=$(grep 'model:' config/config.yaml 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"')
    header
    echo -e "  Teacher: ${GREEN}${cur_model:-unknown}${NC}"
    echo ""
    echo -e "  ${BOLD}Setup${NC}"
    echo "  1) Test API connection"
    echo "  2) Check available models (auto-selects & updates config)"
    echo ""
    echo -e "  ${BOLD}Training${NC}"
    echo "  3) Run training"
    echo "  4) Run evaluation       (single checkpoint, val split)"
    echo "  5) Compare all 3 models (original | teacher | distilled) + graph"
    echo ""
    echo -e "  ${BOLD}Cache${NC}"
    echo "  6) View cached teacher responses"
    echo "  7) Enter manual answers for teacher prompts"
    echo "  8) Reset teacher cache (delete all cached responses)"
    echo ""
    echo -e "  ${BOLD}Docker${NC}"
    echo "  9) Build Docker image"
    echo "  10) Open shell inside container"
    echo ""
    echo -e "  ${BOLD}GPU Setup${NC}"
    echo "  11) Install nvidia-container-toolkit (requires sudo)"
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
    local cache_dir="cache/teacher_logprobs"
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
            run_cmd "sudo docker compose run --rm train python scripts/test_api.py"
            ;;
        2)
            header
            echo -e "${YELLOW}▶ Checking available teacher models...${NC}"
            echo ""
            sudo docker compose run --rm train python scripts/check_models.py
            check_code=$?
            echo ""
            if [ $check_code -eq 0 ]; then
                new_model=$(grep 'model:' config/config.yaml 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"')
                echo -e "  Active model: ${GREEN}${new_model}${NC}"
                echo ""
                echo -n "  Run training now with this model? (y/n): "
                read -r ans
                if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
                    echo ""
                    run_cmd "sudo docker compose run --rm train python train.py"
                else
                    echo ""
                    read -rp "Press Enter to return to menu..."
                fi
            else
                read -rp "Press Enter to return to menu..."
            fi
            ;;
        3)
            header
            cur_model=$(grep 'model:' config/config.yaml 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"')
            local_path=$(grep 'local_model_path' config/config.yaml 2>/dev/null | awk '{print $2}')
            local_path=${local_path:-cache/teacher-model}
            local_exists=""
            [ -d "$local_path" ] && local_exists=" ${GREEN}✓ found${NC}" || local_exists=" ${RED}✗ not found${NC}"
            echo -e "  ${BOLD}Choose teacher source:${NC}"
            echo ""
            echo -e "  1) API teacher    ${CYAN}(${cur_model})${NC}"
            echo -e "  2) Local model    ${CYAN}(${local_path})${NC}${local_exists}"
            echo "  3) Offline        (use cached responses only, no new requests)"
            echo ""
            echo -n "  Select: "
            read -r tsrc
            echo ""
            case $tsrc in
                1)
                    run_cmd "sudo docker compose run --rm train python train.py"
                    ;;
                2)
                    run_cmd "sudo docker compose run --rm train python train.py --local"
                    ;;
                3)
                    run_cmd "sudo docker compose run --rm train python train.py --offline"
                    ;;
                *)
                    echo -e "${RED}Invalid option.${NC}"
                    read -rp "Press Enter to return to menu..."
                    ;;
            esac
            ;;
        4)
            header
            latest=$(find outputs -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -1)
            if [ -z "$latest" ]; then
                echo -e "${RED}No checkpoints found in outputs/. Run training first.${NC}"
                echo ""
                read -rp "Press Enter to return to menu..."
            else
                echo -e "Available checkpoints:"
                find outputs -maxdepth 1 -mindepth 1 -type d | sort | nl -w2 -s') '
                echo ""
                echo -e "Latest: ${GREEN}$latest${NC}"
                echo -n "Enter checkpoint path (or press Enter to use latest): "
                read -r ckpt
                [ -z "$ckpt" ] && ckpt="$latest"
                echo -n "Verbose output? (y/n) [default: n]: "
                read -r vb
                vflag=""
                [ "$vb" = "y" ] || [ "$vb" = "Y" ] && vflag=" --verbose"
                run_cmd "sudo docker compose run --rm evaluate python evaluate.py --checkpoint /workspace/$ckpt$vflag"
            fi
            ;;
        5)
            header
            echo -e "  ${BOLD}Compare all 3 models on LeetCode test split${NC}"
            echo ""
            echo -n "  Number of test problems [default: 30]: "
            read -r np
            [ -z "$np" ] && np=30
            echo -e "  Difficulty filter:"
            echo "  1) All"
            echo "  2) Easy only"
            echo "  3) Medium only"
            echo "  4) Hard only"
            echo -n "  Select [default: 1]: "
            read -r diffchoice
            case $diffchoice in
                2) diff_flag="--difficulty easy" ;;
                3) diff_flag="--difficulty medium" ;;
                4) diff_flag="--difficulty hard" ;;
                *) diff_flag="--difficulty all" ;;
            esac
            echo -n "  Skip teacher evaluation? (y/n) [default: n]: "
            read -r skipteach
            skip_flag=""
            [ "$skipteach" = "y" ] || [ "$skipteach" = "Y" ] && skip_flag="--skip-teacher"
            model_flag=""
            if [ "$skipteach" != "y" ] && [ "$skipteach" != "Y" ]; then
                cur_teacher=$(grep 'model:' config/config.yaml 2>/dev/null | head -1 | awk '{print $2}' | tr -d '"')
                echo ""
                local_teacher_path
                local_teacher_path=$(grep 'local_model_path' config/config.yaml 2>/dev/null | awk '{print $2}')
                local_teacher_path=${local_teacher_path:-cache/teacher-model}
                echo -e "  ${BOLD}Teacher model selection:${NC}"
                echo -e "  1) ${cur_teacher}  ${CYAN}(from config)${NC}"
                echo -e "  2) Local Qwen 14B  ${CYAN}(${local_teacher_path})${NC}"
                echo "  3) konektika-pro"
                echo "  4) konektika-thinking"
                echo "  5) deepseek/deepseek-r1"
                echo "  6) qwen/qwen-2.5-coder-32b-instruct"
                echo "  7) google/gemini-2.5-flash"
                echo -n "  Select [default: 1]: "
                read -r mchoice
                case $mchoice in
                    2) model_flag="--local-teacher" ;;
                    3) model_flag="--teacher-model konektika-pro" ;;
                    4) model_flag="--teacher-model konektika-thinking" ;;
                    5) model_flag="--teacher-model deepseek/deepseek-r1" ;;
                    6) model_flag="--teacher-model qwen/qwen-2.5-coder-32b-instruct" ;;
                    7) model_flag="--teacher-model google/gemini-2.5-flash" ;;
                    *) model_flag="" ;;
                esac
            fi
            run_cmd "sudo docker compose run --rm compare_eval python compare_eval.py --num-problems $np $diff_flag $skip_flag $model_flag"
            echo -e "  Graph saved to: ${GREEN}outputs/eval/comparison.png${NC}"
            ;;
        6)
            view_cache
            ;;
        7)
            header
            echo -e "${YELLOW}▶ Manual answer entry for teacher prompts...${NC}"
            echo ""
            sudo docker compose run --rm -it train python scripts/manual_cache.py
            echo ""
            read -rp "Press Enter to return to menu..."
            ;;
        8)
            header
            cache_dir=$(grep 'teacher_cache_dir' config/config.yaml 2>/dev/null | awk '{print $2}')
            cache_dir=${cache_dir:-cache/teacher_logprobs}
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
        9)
            header
            run_cmd "sudo docker compose build"
            ;;
        10)
            header
            echo -e "${YELLOW}▶ Opening shell inside train container...${NC}"
            echo ""
            sudo docker compose run --rm train bash
            echo ""
            read -rp "Press Enter to return to menu..."
            ;;
        11)
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
