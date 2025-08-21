#!/bin/bash

# Socket/Connection Failure Injection Script
# This script provides an easy interface for socket failure injection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/socket_config.json"
LOG_DIR="${SCRIPT_DIR}/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] ACTION"
    echo ""
    echo "Actions:"
    echo "  start     Start socket failure injection"
    echo "  stop      Stop socket failure injection"
    echo "  status    Show injection status"
    echo "  logs      Show injection logs"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE     Configuration file (default: socket_config.json)"
    echo "  -t, --target NAME     Target container name"
    echo "  -p, --ports PORTS     Target ports (comma-separated, e.g., 80,443,8080)"
    echo "  -i, --ips IPS         Target IPs (comma-separated)"
    echo "  -d, --duration SEC    Duration in seconds"
    echo "  -f, --pattern PATTERN Failure pattern (complete, selective, intermittent)"
    echo "  --protocols PROTOS    Protocols (comma-separated, e.g., tcp,udp)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start                           # Start with default config"
    echo "  $0 -p 80,443 -d 180 start         # Block ports 80,443 for 3 minutes"
    echo "  $0 -f complete start              # Complete network failure"
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
    if [[ -n "$TARGET_PORTS" ]]; then
        # Convert comma-separated ports to JSON array
        local ports_json=$(echo "$TARGET_PORTS" | sed 's/,/", "/g' | sed 's/^/["/' | sed 's/$/"]/')
        jq --argjson ports "$ports_json" '.target_ports = ($ports | map(tonumber))' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$TARGET_IPS" ]]; then
        # Convert comma-separated IPs to JSON array
        local ips_json=$(echo "$TARGET_IPS" | sed 's/,/", "/g' | sed 's/^/["/' | sed 's/$/"]/')
        jq --argjson ips "$ips_json" '.target_ips = $ips' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$DURATION" ]]; then
        jq --argjson duration "$DURATION" '.duration = $duration' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$FAILURE_PATTERN" ]]; then
        jq --arg pattern "$FAILURE_PATTERN" '.failure_pattern = $pattern' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$PROTOCOLS" ]]; then
        # Convert comma-separated protocols to JSON array
        local protocols_json=$(echo "$PROTOCOLS" | sed 's/,/", "/g' | sed 's/^/["/' | sed 's/$/"]/')
        jq --argjson protocols "$protocols_json" '.protocols = $protocols' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    if [[ -n "$TARGET_CONTAINER" ]]; then
        jq --arg target "$TARGET_CONTAINER" '.target_container = $target' "$temp_config" > "${temp_config}.tmp" && mv "${temp_config}.tmp" "$temp_config"
    fi
    
    echo "$temp_config"
}

# Function to start socket failure injection
start_injection() {
    echo "Starting socket failure injection..."
    
    # Check if we have iptables permissions
    if ! command -v iptables &> /dev/null; then
        echo "Error: iptables not found"
        exit 1
    fi
    
    if [[ -z "$TARGET_CONTAINER" ]]; then
        # Check host iptables permissions
        if ! iptables -L &> /dev/null; then
            echo "Error: No permission to run iptables. Run as root or with sudo."
            exit 1
        fi
    fi
    
    # Show current network connections
    echo "Current network connections:"
    if command -v ss &> /dev/null; then
        ss -tuln | head -10
    elif command -v netstat &> /dev/null; then
        netstat -tuln | head -10
    fi
    
    # Update configuration if needed
    local working_config="$CONFIG_FILE"
    if [[ -n "$TARGET_PORTS" || -n "$TARGET_IPS" || -n "$DURATION" || -n "$FAILURE_PATTERN" || -n "$PROTOCOLS" || -n "$TARGET_CONTAINER" ]]; then
        working_config=$(update_config "$CONFIG_FILE")
        echo "Using updated configuration:"
        jq '.' "$working_config"
    fi
    
    # Generate log filename with timestamp
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/socket_failure_${timestamp}.json"
    
    # Run socket failure injection
    echo "Running socket failure injection..."
    python3 "$SCRIPT_DIR/socket_failure.py" \
        --config "$working_config" \
        --action start \
        --log-file "$log_file"
    
    # Clean up temporary config
    if [[ "$working_config" != "$CONFIG_FILE" ]]; then
        rm -f "$working_config"
    fi
    
    echo "Socket failure injection started. Log file: $log_file"
}

# Function to stop socket failure injection
stop_injection() {
    echo "Stopping socket failure injection..."
    
    # Stop Python processes
    if pgrep -f "socket_failure.py" > /dev/null; then
        echo "Stopping socket failure processes..."
        pkill -f "socket_failure.py" || true
    fi
    
    # Clean up iptables rules (emergency cleanup)
    echo "Cleaning up iptables rules..."
    
    # Remove common DROP rules that might have been added
    iptables -D INPUT -p tcp -j DROP 2>/dev/null || true
    iptables -D OUTPUT -p tcp -j DROP 2>/dev/null || true
    iptables -D INPUT -p udp -j DROP 2>/dev/null || true
    iptables -D OUTPUT -p udp -j DROP 2>/dev/null || true
    
    # Try to flush custom chains if they exist
    iptables -F 2>/dev/null || true
    
    echo "Socket failure injection stopped"
}

# Function to show status
show_status() {
    echo "Socket Failure Injection Status:"
    echo "================================"
    
    # Check Python processes
    if pgrep -f "socket_failure.py" > /dev/null; then
        echo "Socket failure processes: Running"
        pgrep -f "socket_failure.py" | while read pid; do
            echo "  PID: $pid"
        done
    else
        echo "Socket failure processes: Not running"
    fi
    
    # Show current iptables rules
    echo ""
    echo "Current iptables rules:"
    if command -v iptables &> /dev/null && iptables -L &> /dev/null; then
        echo "INPUT chain:"
        iptables -L INPUT -n --line-numbers | head -10
        echo ""
        echo "OUTPUT chain:"
        iptables -L OUTPUT -n --line-numbers | head -10
    else
        echo "Cannot access iptables rules"
    fi
    
    # Show current network connections
    echo ""
    echo "Current Network Connections:"
    if command -v ss &> /dev/null; then
        echo "Listening ports:"
        ss -tuln | head -10
        echo ""
        echo "Established connections:"
        ss -tun state established | head -10
    elif command -v netstat &> /dev/null; then
        echo "Network statistics:"
        netstat -tuln | head -10
    else
        echo "Network connection information not available"
    fi
}

# Function to show logs
show_logs() {
    echo "Recent socket failure injection logs:"
    echo "===================================="
    
    if [[ -d "$LOG_DIR" ]]; then
        # Show latest log files
        find "$LOG_DIR" -name "socket_failure_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -5 | while read timestamp file; do
            echo "Log file: $(basename "$file")"
            if command -v jq &> /dev/null; then
                echo "  Start time: $(jq -r '.start_time // "N/A"' "$file")"
                echo "  Data points: $(jq -r '.monitoring_data | length' "$file")"
                echo "  Pattern: $(jq -r '.config.failure_pattern // "N/A"' "$file")"
                echo "  Target ports: $(jq -r '.config.target_ports // [] | join(",")' "$file")"
            fi
        done
        
        # Show latest log content if jq is available
        local latest_log=$(find "$LOG_DIR" -name "socket_failure_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2)
        if [[ -n "$latest_log" && -f "$latest_log" ]] && command -v jq &> /dev/null; then
            echo ""
            echo "Latest log summary:"
            jq -r '.monitoring_data[-5:] | .[] | "\(.timestamp): \(.connection_stats.total) connections, \(.connection_stats.established) established"' "$latest_log" 2>/dev/null || echo "No monitoring data available"
        fi
    else
        echo "No log directory found"
    fi
}

# Parse command line arguments
TARGET_PORTS=""
TARGET_IPS=""
DURATION=""
FAILURE_PATTERN=""
PROTOCOLS=""
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
        -p|--ports)
            TARGET_PORTS="$2"
            shift 2
            ;;
        -i|--ips)
            TARGET_IPS="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION="$2"
            shift 2
            ;;
        -f|--pattern)
            FAILURE_PATTERN="$2"
            shift 2
            ;;
        --protocols)
            PROTOCOLS="$2"
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