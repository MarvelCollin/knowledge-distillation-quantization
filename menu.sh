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
    echo "  8) Check available models (auto-selects & updates config)"
    echo ""
    echo -e "  ${BOLD}Training${NC}"
    echo "  2) Run training (fetch teacher responses + train)"
    echo "  2o) Run training offline (use cached data only, no API calls)"
    echo "  3) Run evaluation"
    echo ""
    echo -e "  ${BOLD}Cache${NC}"
    echo "  7) View cached teacher responses"
    echo ""
    echo -e "  ${BOLD}Docker${NC}"
    echo "  4) Build Docker image"
    echo "  5) Open shell inside container"
    echo ""
    echo -e "  ${BOLD}GPU Setup${NC}"
    echo "  6) Install nvidia-container-toolkit (requires sudo)"
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

while true; do
    show_menu
    read -r choice
    case $choice in
        1)
            header
            run_cmd "sudo docker compose run --rm train python scripts/test_api.py"
            ;;
        2)
            header
            run_cmd "sudo docker compose run --rm train python train.py"
            ;;
        2o|2O)
            header
            run_cmd "sudo docker compose run --rm train python train.py --offline"
            ;;
        3)
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
                run_cmd "sudo docker compose run --rm evaluate python evaluate.py --checkpoint /workspace/$ckpt"
            fi
            ;;
        4)
            header
            run_cmd "sudo docker compose build"
            ;;
        5)
            header
            echo -e "${YELLOW}▶ Opening shell inside train container...${NC}"
            echo ""
            sudo docker compose run --rm train bash
            echo ""
            read -rp "Press Enter to return to menu..."
            ;;
        6)
            install_nvidia_toolkit
            ;;
        7)
            view_cache
            ;;
        8)
            header
            echo -e "${YELLOW}▶ Checking available teacher models...${NC}"
            echo ""
            sudo docker compose run --rm train python scripts/check_models.py
            check_code=$?
            echo ""
            if [ $check_code -eq 0 ]; then
                local new_model
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
