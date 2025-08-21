#!/bin/bash

# Disk I/O Stress Injection Script
# This script provides an easy interface for disk I/O stress injection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/disk_config.json"
LOG_DIR="${SCRIPT_DIR}/logs"
CONTAINER_NAME="disk-stress-injector"

# Create logs directory
mkdir -p "$LOG_DIR"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] ACTION"
    echo ""
    echo "Actions:"
    echo "  start     Start disk I/O stress injection"
    echo "  stop      Stop disk I/O stress injection"
    echo "  status    Show injection status"
    echo "  logs      Show injection logs"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE     Configuration file (default: disk_config.json)"
    echo "  -t, --target NAME     Target container name"
    echo "  -s, --size SIZE       File size (e.g., 1G, 512M)"
    echo "  -d, --duration SEC    Duration in seconds"
    echo "  -w, --workers NUM     Number of I/O workers"
    echo "  -p, --path PATH       Temporary path for I/O operations"
    echo "  -m, --method METHOD   I/O method (sync, dsync, direct)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start                           # Start with default config"
    echo "  $0 -s 2G -d 180 start             # 2GB files for 3 minutes"
    echo "  $0 -w 4 -m sync start             # 4 workers with sync I/O"
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
    if [[ -n "$FILE_SIZE" ]]; then
        jq --arg size "$FILE_SIZE" '.file_size = $size' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$DURATION" ]]; then
        jq --argjson duration "$DURATION" '.duration = $duration' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$WORKERS" ]]; then
        jq --argjson workers "$WORKERS" '.workers = $workers' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$TEMP_PATH" ]]; then
        jq --arg path "$TEMP_PATH" '.temp_path = $path' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$IO_METHOD" ]]; then
        jq --arg method "$IO_METHOD" '.method = $method' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$TARGET_CONTAINER" ]]; then
        jq --arg target "$TARGET_CONTAINER" '.target_container = $target' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    echo "$temp_config"
}

# Function to start disk stress injection
start_injection() {
    echo "Starting disk I/O stress injection..."
    
    # Check available disk space
    if command -v df &> /dev/null; then
        echo "Current disk usage:"
        df -h
    fi
    
    # Update configuration if needed
    local working_config="$CONFIG_FILE"
    if [[ -n "$FILE_SIZE" || -n "$DURATION" || -n "$WORKERS" || -n "$TEMP_PATH" || -n "$IO_METHOD" || -n "$TARGET_CONTAINER" ]]; then
        working_config=$(update_config "$CONFIG_FILE")
        echo "Using updated configuration:"
        jq '.' "$working_config"
    fi
    
    # Generate log filename with timestamp
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/disk_stress_${timestamp}.json"
    
    # Check if running in Docker environment
    if command -v docker &> /dev/null && [[ -n "$TARGET_CONTAINER" ]]; then
        echo "Running disk stress injection in container: $TARGET_CONTAINER"
        
        # Build Docker image if it doesn't exist
        if ! docker image inspect disk-stress-injector &> /dev/null; then
            echo "Building disk stress injector Docker image..."
            docker build -t disk-stress-injector "$SCRIPT_DIR"
        fi
        
        # Run stress injection in Docker
        docker run --rm -d \
            --name "$CONTAINER_NAME" \
            --network container:"$TARGET_CONTAINER" \
            --pid container:"$TARGET_CONTAINER" \
            -v "$working_config":/chaos/config.json \
            -v "$LOG_DIR":/chaos/logs \
            disk-stress-injector \
            python3 /usr/local/bin/disk_stress.py \
            --config /chaos/config.json \
            --action start \
            --log-file "/chaos/logs/disk_stress_${timestamp}.json"
    else
        # Run stress injection on host
        echo "Running disk stress injection on host..."
        python3 "$SCRIPT_DIR/disk_stress.py" \
            --config "$working_config" \
            --action start \
            --log-file "$log_file"
    fi
    
    # Clean up temporary config
    if [[ "$working_config" != "$CONFIG_FILE" ]]; then
        rm -f "$working_config"
    fi
    
    echo "Disk stress injection started. Log file: $log_file"
}

# Function to stop disk stress injection
stop_injection() {
    echo "Stopping disk I/O stress injection..."
    
    # Stop Docker container if running
    if docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"; then
        echo "Stopping Docker container: $CONTAINER_NAME"
        docker stop "$CONTAINER_NAME"
    fi
    
    # Stop host processes
    if pgrep -f "disk_stress.py" > /dev/null; then
        echo "Stopping host disk stress processes..."
        pkill -f "disk_stress.py" || true
    fi
    
    # Stop stress-ng processes
    if pgrep -f "stress-ng.*--hdd" > /dev/null; then
        echo "Stopping stress-ng disk processes..."
        pkill -f "stress-ng.*--hdd" || true
    fi
    
    echo "Disk stress injection stopped"
}

# Function to show status
show_status() {
    echo "Disk I/O Stress Injection Status:"
    echo "================================="
    
    # Check Docker container
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "^${CONTAINER_NAME}"; then
        echo "Docker container: Running"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep "$CONTAINER_NAME"
    else
        echo "Docker container: Not running"
    fi
    
    # Check host processes
    if pgrep -f "disk_stress.py" > /dev/null; then
        echo "Host processes: Running"
        pgrep -f "disk_stress.py" | while read pid; do
            echo "  PID: $pid"
        done
    else
        echo "Host processes: Not running"
    fi
    
    # Check stress-ng processes
    if pgrep -f "stress-ng.*--hdd" > /dev/null; then
        echo "stress-ng disk processes: Running"
        pgrep -f "stress-ng.*--hdd" | while read pid; do
            echo "  PID: $pid"
        done
    else
        echo "stress-ng disk processes: Not running"
    fi
    
    # Show current disk usage
    echo ""
    echo "Current Disk Usage:"
    if command -v df &> /dev/null; then
        df -h
    else
        echo "Disk usage information not available"
    fi
    
    # Show I/O statistics if available
    echo ""
    echo "Current I/O Statistics:"
    if command -v iostat &> /dev/null; then
        iostat -x 1 1 2>/dev/null | tail -n +4 || echo "I/O statistics not available"
    elif [[ -f /proc/diskstats ]]; then
        echo "Recent disk activity:"
        cat /proc/diskstats | head -5
    else
        echo "I/O statistics not available"
    fi
}

# Function to show logs
show_logs() {
    echo "Recent disk stress injection logs:"
    echo "================================="
    
    if [[ -d "$LOG_DIR" ]]; then
        # Show latest log files
        find "$LOG_DIR" -name "disk_stress_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -5 | while read timestamp file; do
            echo "Log file: $(basename "$file")"
            if command -v jq &> /dev/null; then
                echo "  Start time: $(jq -r '.start_time // "N/A"' "$file")"
                echo "  Data points: $(jq -r '.monitoring_data | length' "$file")"
                echo "  File size: $(jq -r '.config.file_size // "N/A"' "$file")"
                echo "  Workers: $(jq -r '.config.workers // "N/A"' "$file")"
            fi
        done
        
        # Show latest log content if jq is available
        local latest_log=$(find "$LOG_DIR" -name "disk_stress_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
        if [[ -n "$latest_log" && -f "$latest_log" ]] && command -v jq &> /dev/null; then
            echo ""
            echo "Latest log summary:"
            jq -r '.monitoring_data[-5:] | .[] | "\(.timestamp): Disk \(.disk_used_percent)%, Free \(.disk_free_gb)GB"' "$latest_log" 2>/dev/null || echo "No monitoring data available"
        fi
    else
        echo "No log directory found"
    fi
}

# Parse command line arguments
FILE_SIZE=""
DURATION=""
WORKERS=""
TEMP_PATH=""
IO_METHOD=""
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
        -s|--size)
            FILE_SIZE="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -p|--path)
            TEMP_PATH="$2"
            shift 2
            ;;
        -m|--method)
            IO_METHOD="$2"
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