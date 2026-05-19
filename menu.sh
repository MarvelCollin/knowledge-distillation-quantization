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

current_model() {
    grep "^  model:" config/config.yaml 2>/dev/null | awk '{print $2}' || echo "unknown"
}

show_menu() {
    header
    echo -e "  Current teacher model: ${GREEN}$(current_model)${NC}"
    echo ""
    echo -e "  ${BOLD}Setup${NC}"
    echo "  1) Find working free model (auto-updates config)"
    echo "  2) Test API connection"
    echo ""
    echo -e "  ${BOLD}Training${NC}"
    echo "  3) Run training"
    echo "  4) Run evaluation"
    echo ""
    echo -e "  ${BOLD}Docker${NC}"
    echo "  5) Build Docker image"
    echo "  6) Open shell inside container"
    echo ""
    echo -e "  ${BOLD}GPU Setup${NC}"
    echo "  7) Install nvidia-container-toolkit (requires sudo)"
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
            run_cmd "sudo docker compose run --rm train python scripts/find_working_model.py"
            ;;
        2)
            header
            run_cmd "sudo docker compose run --rm train python scripts/test_api.py"
            ;;
        3)
            header
            run_cmd "sudo docker compose run --rm train python train.py"
            ;;
        4)
            header
            run_cmd "sudo docker compose run --rm evaluate python evaluate.py"
            ;;
        5)
            header
            run_cmd "sudo docker compose build"
            ;;
        6)
            header
            echo -e "${YELLOW}▶ Opening shell inside train container...${NC}"
            echo ""
            sudo docker compose run --rm train bash
            echo ""
            read -rp "Press Enter to return to menu..."
            ;;
        7)
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
