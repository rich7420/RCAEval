#!/bin/bash

# Locust Traffic Generator Startup Script
# Supports both standalone and distributed modes

set -e

# Default configuration
DEFAULT_APPLICATION="otel_demo"
DEFAULT_MODE="standalone"
DEFAULT_USERS=50
DEFAULT_SPAWN_RATE=5
DEFAULT_RUN_TIME="5m"
DEFAULT_HOST="http://localhost:8080"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Locust Traffic Generator Startup Script

Usage: $0 [OPTIONS]

OPTIONS:
    -a, --application    Target application (otel_demo|online_boutique) [default: $DEFAULT_APPLICATION]
    -m, --mode          Run mode (standalone|distributed|docker) [default: $DEFAULT_MODE]
    -u, --users         Number of users [default: $DEFAULT_USERS]
    -r, --spawn-rate    Spawn rate (users/second) [default: $DEFAULT_SPAWN_RATE]
    -t, --run-time      Run time (e.g., 5m, 1h) [default: $DEFAULT_RUN_TIME]
    -H, --host          Target host URL [default: $DEFAULT_HOST]
    -w, --workers       Number of workers (distributed mode) [default: 2]
    -p, --port          Web UI port [default: 8089]
    --headless          Run in headless mode (no web UI)
    --config            Configuration file [default: config.json]
    -h, --help          Show this help message

EXAMPLES:
    # Start standalone Locust for OTel Demo
    $0 -a otel_demo -u 100 -r 10

    # Start distributed Locust with 3 workers
    $0 -m distributed -w 3 -a online_boutique

    # Start in Docker mode
    $0 -m docker -a otel_demo

    # Run headless test for 10 minutes
    $0 --headless -t 10m -u 200

EOF
}

# Parse command line arguments
APPLICATION="$DEFAULT_APPLICATION"
MODE="$DEFAULT_MODE"
USERS="$DEFAULT_USERS"
SPAWN_RATE="$DEFAULT_SPAWN_RATE"
RUN_TIME="$DEFAULT_RUN_TIME"
HOST="$DEFAULT_HOST"
WORKERS=2
PORT=8089
HEADLESS=false
CONFIG_FILE="config.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--application)
            APPLICATION="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -u|--users)
            USERS="$2"
            shift 2
            ;;
        -r|--spawn-rate)
            SPAWN_RATE="$2"
            shift 2
            ;;
        -t|--run-time)
            RUN_TIME="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate application
if [[ "$APPLICATION" != "otel_demo" && "$APPLICATION" != "online_boutique" ]]; then
    log_error "Invalid application: $APPLICATION. Must be 'otel_demo' or 'online_boutique'"
    exit 1
fi

# Validate mode
if [[ "$MODE" != "standalone" && "$MODE" != "distributed" && "$MODE" != "docker" ]]; then
    log_error "Invalid mode: $MODE. Must be 'standalone', 'distributed', or 'docker'"
    exit 1
fi

# Set application-specific defaults
case $APPLICATION in
    otel_demo)
        LOCUSTFILE="otel_demo_users.py"
        USER_CLASS="OTelDemoUser"
        [[ "$HOST" == "$DEFAULT_HOST" ]] && HOST="http://localhost:8080"
        ;;
    online_boutique)
        LOCUSTFILE="online_boutique_users.py"
        USER_CLASS="OnlineBoutiqueUser"
        [[ "$HOST" == "$DEFAULT_HOST" ]] && HOST="http://localhost:80"
        ;;
esac

# Function to check if Locust is installed
check_locust() {
    if ! command -v locust &> /dev/null; then
        log_error "Locust is not installed. Please install it with: pip install locust"
        log_info "Or install from requirements.txt: pip install -r requirements.txt"
        exit 1
    fi
}

# Function to check if required files exist
check_files() {
    if [[ ! -f "$LOCUSTFILE" ]]; then
        log_error "Locustfile not found: $LOCUSTFILE"
        exit 1
    fi
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_warning "Config file not found: $CONFIG_FILE. Using defaults."
    fi
}

# Function to start standalone Locust
start_standalone() {
    log_info "Starting Locust in standalone mode..."
    log_info "Application: $APPLICATION"
    log_info "Users: $USERS, Spawn rate: $SPAWN_RATE, Run time: $RUN_TIME"
    log_info "Target host: $HOST"
    log_info "Web UI: http://localhost:$PORT"
    
    LOCUST_CMD="locust --locustfile=$LOCUSTFILE --host=$HOST --web-port=$PORT"
    
    if [[ "$HEADLESS" == "true" ]]; then
        LOCUST_CMD="$LOCUST_CMD --headless --users=$USERS --spawn-rate=$SPAWN_RATE --run-time=$RUN_TIME"
        log_info "Running in headless mode"
    fi
    
    log_info "Starting Locust: $LOCUST_CMD"
    exec $LOCUST_CMD
}

# Function to start distributed Locust
start_distributed() {
    log_info "Starting Locust in distributed mode..."
    log_info "Workers: $WORKERS"
    
    if [[ ! -f "distributed_runner.py" ]]; then
        log_error "Distributed runner not found: distributed_runner.py"
        exit 1
    fi
    
    RUNNER_CMD="python distributed_runner.py --mode=distributed --application=$APPLICATION --workers=$WORKERS --users=$USERS --spawn-rate=$SPAWN_RATE --run-time=$RUN_TIME --config=$CONFIG_FILE"
    
    log_info "Starting distributed runner: $RUNNER_CMD"
    exec $RUNNER_CMD
}

# Function to start Docker mode
start_docker() {
    log_info "Starting Locust in Docker mode..."
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
        log_error "Docker or docker-compose is not installed"
        exit 1
    fi
    
    if [[ ! -f "docker-compose.locust.yml" ]]; then
        log_error "Docker compose file not found: docker-compose.locust.yml"
        exit 1
    fi
    
    # Set environment variables for Docker
    export TARGET_HOST="$HOST"
    export BOUTIQUE_HOST="$HOST"
    
    case $APPLICATION in
        otel_demo)
            log_info "Starting OTel Demo Locust cluster..."
            docker-compose -f docker-compose.locust.yml up --build
            ;;
        online_boutique)
            log_info "Starting Online Boutique Locust cluster..."
            docker-compose -f docker-compose.locust.yml --profile boutique up --build
            ;;
    esac
}

# Function to cleanup on exit
cleanup() {
    log_info "Cleaning up..."
    if [[ "$MODE" == "docker" ]]; then
        docker-compose -f docker-compose.locust.yml down
    fi
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Main execution
log_info "Locust Traffic Generator"
log_info "======================="

case $MODE in
    standalone)
        check_locust
        check_files
        start_standalone
        ;;
    distributed)
        check_locust
        check_files
        start_distributed
        ;;
    docker)
        check_files
        start_docker
        ;;
esac