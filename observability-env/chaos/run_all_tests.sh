#!/bin/bash

# Comprehensive Test Runner for Chaos Engineering Modules
# This script runs all tests for the chaos engineering capabilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Function to run a test module
run_test_module() {
    local module_name=$1
    local test_script=$2
    local test_dir=$3
    
    print_status "INFO" "Running $module_name tests..."
    echo "=" * 60
    
    cd "$SCRIPT_DIR/$test_dir"
    
    if [[ ! -f "$test_script" ]]; then
        print_status "ERROR" "Test script not found: $test_script"
        return 1
    fi
    
    if python3 "$test_script"; then
        print_status "SUCCESS" "$module_name tests passed"
        ((PASSED_TESTS++))
        return 0
    else
        print_status "ERROR" "$module_name tests failed"
        ((FAILED_TESTS++))
        return 1
    fi
}

# Function to check dependencies
check_dependencies() {
    print_status "INFO" "Checking dependencies..."
    
    # Check Python 3
    if ! command -v python3 &> /dev/null; then
        print_status "ERROR" "Python 3 is required but not installed"
        return 1
    fi
    
    # Check required Python packages
    python3 -c "import psutil" 2>/dev/null || {
        print_status "WARNING" "psutil package not found. Some tests may fail."
        print_status "INFO" "Install with: pip3 install psutil"
    }
    
    # Check stress-ng (optional)
    if ! command -v stress-ng &> /dev/null; then
        print_status "WARNING" "stress-ng not found. Stress injection tests will be skipped."
        print_status "INFO" "Install with: sudo apt-get install stress-ng (Ubuntu/Debian)"
    fi
    
    # Check tc (traffic control) - optional
    if ! command -v tc &> /dev/null; then
        print_status "WARNING" "tc (traffic control) not found. Network chaos tests may fail."
        print_status "INFO" "Install with: sudo apt-get install iproute2"
    fi
    
    # Check iptables - optional
    if ! command -v iptables &> /dev/null; then
        print_status "WARNING" "iptables not found. Socket failure tests may fail."
        print_status "INFO" "Install with: sudo apt-get install iptables"
    fi
    
    # Check jq - optional
    if ! command -v jq &> /dev/null; then
        print_status "WARNING" "jq not found. Some shell script features may not work."
        print_status "INFO" "Install with: sudo apt-get install jq"
    fi
    
    print_status "SUCCESS" "Dependency check completed"
    return 0
}

# Function to run configuration validation tests
test_configurations() {
    print_status "INFO" "Testing configuration files..."
    
    local config_files=(
        "cpu/cpu_config.json"
        "memory/memory_config.json"
        "disk/disk_config.json"
        "network/delay_config.json"
        "network/loss_config.json"
        "socket/socket_config.json"
    )
    
    for config_file in "${config_files[@]}"; do
        local full_path="$SCRIPT_DIR/$config_file"
        
        if [[ ! -f "$full_path" ]]; then
            print_status "ERROR" "Configuration file missing: $config_file"
            return 1
        fi
        
        # Validate JSON syntax
        if command -v jq &> /dev/null; then
            if ! jq empty "$full_path" 2>/dev/null; then
                print_status "ERROR" "Invalid JSON in configuration file: $config_file"
                return 1
            fi
        else
            # Fallback validation using Python
            if ! python3 -c "import json; json.load(open('$full_path'))" 2>/dev/null; then
                print_status "ERROR" "Invalid JSON in configuration file: $config_file"
                return 1
            fi
        fi
        
        print_status "SUCCESS" "Configuration file valid: $config_file"
    done
    
    return 0
}

# Function to test shell scripts
test_shell_scripts() {
    print_status "INFO" "Testing shell scripts..."
    
    local shell_scripts=(
        "cpu/inject_cpu_stress.sh"
        "memory/inject_memory_stress.sh"
        "disk/inject_disk_stress.sh"
        "socket/inject_socket_failure.sh"
    )
    
    for script in "${shell_scripts[@]}"; do
        local full_path="$SCRIPT_DIR/$script"
        
        if [[ ! -f "$full_path" ]]; then
            print_status "ERROR" "Shell script missing: $script"
            return 1
        fi
        
        if [[ ! -x "$full_path" ]]; then
            print_status "ERROR" "Shell script not executable: $script"
            return 1
        fi
        
        # Test help option
        if "$full_path" --help &>/dev/null; then
            print_status "SUCCESS" "Shell script working: $script"
        else
            print_status "WARNING" "Shell script help option failed: $script"
        fi
    done
    
    return 0
}

# Function to run integration tests
run_integration_tests() {
    print_status "INFO" "Running integration tests..."
    
    cd "$SCRIPT_DIR"
    
    if python3 test_chaos_suite.py; then
        print_status "SUCCESS" "Integration tests passed"
        return 0
    else
        print_status "ERROR" "Integration tests failed"
        return 1
    fi
}

# Function to generate test report
generate_report() {
    local total=$((PASSED_TESTS + FAILED_TESTS))
    
    echo ""
    echo "=" * 60
    print_status "INFO" "CHAOS ENGINEERING TEST REPORT"
    echo "=" * 60
    
    echo "Total test modules: $total"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    
    if [[ $FAILED_TESTS -eq 0 ]]; then
        print_status "SUCCESS" "All tests passed! ✓"
        echo ""
        print_status "INFO" "Chaos engineering modules are ready for use."
        return 0
    else
        print_status "ERROR" "Some tests failed! ✗"
        echo ""
        print_status "INFO" "Please check the failed tests and resolve issues."
        return 1
    fi
}

# Main execution
main() {
    echo "Chaos Engineering Module Test Suite"
    echo "=" * 60
    echo "Testing all chaos engineering capabilities..."
    echo ""
    
    # Check dependencies first
    if ! check_dependencies; then
        print_status "ERROR" "Dependency check failed"
        exit 1
    fi
    
    echo ""
    
    # Test configuration files
    if ! test_configurations; then
        print_status "ERROR" "Configuration validation failed"
        exit 1
    fi
    
    echo ""
    
    # Test shell scripts
    if ! test_shell_scripts; then
        print_status "ERROR" "Shell script validation failed"
        exit 1
    fi
    
    echo ""
    
    # Run individual module tests
    print_status "INFO" "Running individual module tests..."
    echo ""
    
    # CPU stress tests
    run_test_module "CPU Stress" "test_cpu_stress.py" "cpu"
    ((TOTAL_TESTS++))
    
    echo ""
    
    # Memory stress tests
    run_test_module "Memory Stress" "test_memory_stress.py" "memory"
    ((TOTAL_TESTS++))
    
    echo ""
    
    # Disk stress tests
    run_test_module "Disk Stress" "test_disk_stress.py" "disk"
    ((TOTAL_TESTS++))
    
    echo ""
    
    # Network chaos tests
    run_test_module "Network Chaos" "test_network_chaos.py" "network"
    ((TOTAL_TESTS++))
    
    echo ""
    
    # Socket failure tests
    run_test_module "Socket Failure" "test_socket_failure.py" "socket"
    ((TOTAL_TESTS++))
    
    echo ""
    
    # Integration tests
    if run_integration_tests; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    
    # Generate final report
    generate_report
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --quick        Run only basic validation tests"
        echo "  --verbose      Enable verbose output"
        echo ""
        echo "This script runs comprehensive tests for all chaos engineering modules."
        exit 0
        ;;
    --quick)
        print_status "INFO" "Running quick validation tests only..."
        check_dependencies
        test_configurations
        test_shell_scripts
        print_status "SUCCESS" "Quick validation completed"
        exit 0
        ;;
    --verbose)
        set -x
        main
        ;;
    "")
        main
        ;;
    *)
        print_status "ERROR" "Unknown option: $1"
        print_status "INFO" "Use --help for usage information"
        exit 1
        ;;
esac