#!/bin/bash

# CPU Stress Injection Script
# This script provides an easy interface for CPU stress injection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/cpu_config.json"
LOG_DIR="${SCRIPT_DIR}/logs"
CONTAINER_NAME="cpu-stress-injector"

# Create logs directory
mkdir -p "$LOG_DIR"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] ACTION"
    echo ""
    echo "Actions:"
    echo "  start     Start CPU stress injection"
    echo "  stop      Stop CPU stress injection"
    echo "  status    Show injection status"
    echo "  logs      Show injection logs"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE     Configuration file (default: cpu_config.json)"
    echo "  -t, --target NAME     Target container name"
    echo "  -i, --intensity VAL   CPU intensity (e.g., 80% or 4)"
    echo "  -d, --duration SEC    Duration in seconds"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start                           # Start with default config"
    echo "  $0 -i 50% -d 180 start            # 50% CPU for 3 minutes"
    echo "  $0 -t checkoutservice start       # Target specific container"
    echo "  $0 stop                           # Stop injection"
}

# Function to update config
update_config() {
    local config_file="$1"
    local temp_config=$(mktemp)
    
    # Copy original config
    cp "$config_file" "$temp_config"
    
    # Update values if provided
    if [[ -n "$INTENSITY" ]]; then
        jq --arg intensity "$INTENSITY" '.intensity = $intensity' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$DURATION" ]]; then
        jq --argjson duration "$DURATION" '.duration = $duration' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$TARGET_CONTAINER" ]]; then
        jq --arg target "$TARGET_CONTAINER" '.target_container = $target' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    echo "$temp_config"
}

# Function to start CPU stress injection
start_injection() {
    echo "Starting CPU stress injection..."
    
    # Update configuration if needed
    local working_config="$CONFIG_FILE"
    if [[ -n "$INTENSITY" || -n "$DURATION" || -n "$TARGET_CONTAINER" ]]; then
        working_config=$(update_config "$CONFIG_FILE")
        echo "Using updated configuration:"
        jq '.' "$working_config"
    fi
    
    # Generate log filename with timestamp
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/cpu_stress_${timestamp}.json"
    
    # Check if running in Docker environment
    if command -v docker &> /dev/null && [[ -n "$TARGET_CONTAINER" ]]; then
        echo "Running CPU stress injection in container: $TARGET_CONTAINER"
        
        # Build Docker image if it doesn't exist
        if ! docker image inspect cpu-stress-injector &> /dev/null; then
            echo "Building CPU stress injector Docker image..."
            docker build -t cpu-stress-injector "$SCRIPT_DIR"
        fi
        
        # Run stress injection in Docker
        docker run --rm -d \
            --name "$CONTAINER_NAME" \
            --network container:"$TARGET_CONTAINER" \
            --pid container:"$TARGET_CONTAINER" \
            -v "$working_config":/chaos/config.json \
            -v "$LOG_DIR":/chaos/logs \
            cpu-stress-injector \
            python3 /usr/local/bin/cpu_stress.py \
            --config /chaos/config.json \
            --action start \
            --log-file "/chaos/logs/cpu_stress_${timestamp}.json"
    else
        # Run stress injection on host
        echo "Running CPU stress injection on host..."
        python3 "$SCRIPT_DIR/cpu_stress.py" \
            --config "$working_config" \
            --action start \
            --log-file "$log_file"
    fi
    
    # Clean up temporary config
    if [[ "$working_config" != "$CONFIG_FILE" ]]; then
        rm -f "$working_config"
    fi
    
    echo "CPU stress injection started. Log file: $log_file"
}

# Function to stop CPU stress injection
stop_injection() {
    echo "Stopping CPU stress injection..."
    
    # Stop Docker container if running
    if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping Docker container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME"
    fi
    
    # Stop host processes
    if pgrep -f "cpu_stress.py" > /dev/null; then
        echo "Stopping host CPU stress processes..."
        pkill -f "cpu_stress.py" || true
    fi
    
    # Stop stress-ng processes
    if pgrep stress-ng > /dev/null; then
        echo "Stopping stress-ng processes..."
        pkill stress-ng || true
    fi
    
    echo "CPU stress injection stopped"
}

# Function to show status
show_status() {
    echo "CPU Stress Injection Status:"
    echo "============================"
    
    # Check Docker container
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "^${CONTAINER_NAME}"; then
        echo "Docker container: Running"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep "$CONTAINER_NAME"
    else
        echo "Docker container: Not running"
    fi
    
    # Check host processes
    if pgrep -f "cpu_stress.py" > /dev/null; then
        echo "Host processes: Running"
        pgrep -f "cpu_stress.py" | while read pid; do
            echo "  PID: $pid"
        done
    else
        echo "Host processes: Not running"
    fi
    
    # Check stress-ng processes
    if pgrep stress-ng > /dev/null; then
        echo "stress-ng processes: Running"
        pgrep stress-ng | while read pid; do
            echo "  PID: $pid"
        done
    else
        echo "stress-ng processes: Not running"
    fi
    
    # Show current CPU usage
    echo ""
    echo "Current CPU Usage:"
    if command -v top &> /dev/null; then
        top -bn1 | grep "Cpu(s)" || echo "CPU usage information not available"
    fi
}

# Function to show logs
show_logs() {
    echo "Recent CPU stress injection logs:"
    echo "================================"
    
    if [[ -d "$LOG_DIR" ]]; then
        # Show latest log files
        find "$LOG_DIR" -name "cpu_stress_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -5 | while read timestamp file; do
            echo "Log file: $(basename "$file")"
            if command -v jq &> /dev/null; then
                echo "  Start time: $(jq -r '.start_time // "N/A"' "$file")"
                echo "  Data points: $(jq -r '.monitoring_data | length' "$file")"
            fi
        done
        
        # Show latest log content if jq is available
        local latest_log=$(find "$LOG_DIR" -name "cpu_stress_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
        if [[ -n "$latest_log" && -f "$latest_log" ]] && command -v jq &> /dev/null; then
            echo ""
            echo "Latest log summary:"
            jq -r '.monitoring_data[-5:] | .[] | "\(.timestamp): CPU \(.cpu_percent)%, Memory \(.memory_percent)%"' "$latest_log" 2>/dev/null || echo "No monitoring data available"
        fi
    else
        echo "No log directory found"
    fi
}

# Parse command line arguments
INTENSITY=""
DURATION=""
TARGET_CONTAINER=""
ACTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -t|--target)
            TARGET_CONTAINER="$2"
            shift 2
            ;;
        -i|--intensity)
            INTENSITY="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        start|stop|status|logs)
            ACTION="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate action
if [[ -z "$ACTION" ]]; then
    echo "Error: No action specified"
    show_usage
    exit 1
fi

# Check dependencies
if ! command -v jq &> /dev/null; then
    echo "Warning: jq not found. Some features may not work properly."
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Execute action
case $ACTION in
    start)
        start_injection
        ;;
    stop)
        stop_injection
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Unknown action: $ACTION"
        show_usage
        exit 1
        ;;
esac