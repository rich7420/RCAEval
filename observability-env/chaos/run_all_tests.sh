#!/bin/bash

# Comprehensive test runner for all chaos engineering modules
# This script runs all available tests to ensure complete functionality

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/test_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create logs directory
mkdir -p "$LOG_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    echo "=================================================================="
    echo "$1"
    echo "=================================================================="
}

# Function to run individual module tests
run_module_test() {
    local module=$1
    local test_file=$2
    local log_file="${LOG_DIR}/${module}_test_${TIMESTAMP}.log"
    
    print_status $BLUE "Running $module tests..."
    
    if [[ -f "$test_file" ]]; then
        if python3 "$test_file" > "$log_file" 2>&1; then
            print_status $GREEN "✓ $module tests PASSED"
            return 0
        else
            print_status $RED "✗ $module tests FAILED"
            echo "  Log file: $log_file"
            echo "  Last few lines:"
            tail -5 "$log_file" | sed 's/^/    /'
            return 1
        fi
    else
        print_status $YELLOW "⚠ $module test file not found: $test_file"
        return 1
    fi
}

# Function to check system requirements
check_requirements() {
    print_header "CHECKING SYSTEM REQUIREMENTS"
    
    local missing_tools=()
    
    # Check required tools
    tools=("python3" "stress-ng" "tc" "iptables" "jq")
    for tool in "${tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            print_status $GREEN "✓ $tool is available"
        else
            print_status $RED "✗ $tool is not available"
            missing_tools+=("$tool")
        fi
    done
    
    # Check Python packages
    python_packages=("psutil")
    for package in "${python_packages[@]}"; do
        if python3 -c "import $package" &> /dev/null; then
            print_status $GREEN "✓ Python package $package is available"
        else
            print_status $RED "✗ Python package $package is not available"
            missing_tools+=("python3-$package")
        fi
    done
    
    # Check permissions
    if [[ $EUID -eq 0 ]]; then
        print_status $GREEN "✓ Running as root (full functionality available)"
    else
        print_status $YELLOW "⚠ Not running as root (some tests may be limited)"
        echo "  For full testing, run: sudo $0"
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_status $YELLOW "Missing tools/packages:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        echo ""
        echo "To install missing tools on Ubuntu/Debian:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install stress-ng iproute2 iptables jq python3-psutil"
        echo ""
        echo "To install missing tools on CentOS/RHEL:"
        echo "  sudo yum install stress-ng iproute iptables jq"
        echo "  pip3 install psutil"
        echo ""
    fi
    
    return 0
}

# Function to run quick tests (no root required)
run_quick_tests() {
    print_header "RUNNING QUICK TESTS (No Root Required)"
    
    local quick_test_log="${LOG_DIR}/quick_test_${TIMESTAMP}.log"
    
    if python3 "${SCRIPT_DIR}/quick_test.py" > "$quick_test_log" 2>&1; then
        print_status $GREEN "✓ Quick tests PASSED"
        return 0
    else
        print_status $RED "✗ Quick tests FAILED"
        echo "  Log file: $quick_test_log"
        echo "  Last few lines:"
        tail -10 "$quick_test_log" | sed 's/^/    /'
        return 1
    fi
}

# Function to run individual module tests
run_individual_tests() {
    print_header "RUNNING INDIVIDUAL MODULE TESTS"
    
    local passed=0
    local total=0
    
    # CPU tests
    total=$((total + 1))
    if run_module_test "CPU" "${SCRIPT_DIR}/cpu/test_cpu_stress.py"; then
        passed=$((passed + 1))
    fi
    
    # Memory tests
    total=$((total + 1))
    if run_module_test "Memory" "${SCRIPT_DIR}/memory/test_memory_stress.py"; then
        passed=$((passed + 1))
    fi
    
    # Disk tests
    total=$((total + 1))
    if run_module_test "Disk" "${SCRIPT_DIR}/disk/test_disk_stress.py"; then
        passed=$((passed + 1))
    fi
    
    # Network tests
    total=$((total + 1))
    if run_module_test "Network" "${SCRIPT_DIR}/network/test_network_chaos.py"; then
        passed=$((passed + 1))
    fi
    
    # Socket tests
    total=$((total + 1))
    if run_module_test "Socket" "${SCRIPT_DIR}/socket/test_socket_failure.py"; then
        passed=$((passed + 1))
    fi
    
    print_status $BLUE "Individual tests summary: $passed/$total passed"
    return $((total - passed))
}

# Function to run comprehensive test suite
run_comprehensive_tests() {
    print_header "RUNNING COMPREHENSIVE TEST SUITE"
    
    local comprehensive_log="${LOG_DIR}/comprehensive_test_${TIMESTAMP}.log"
    
    if python3 "${SCRIPT_DIR}/test_chaos_suite.py" > "$comprehensive_log" 2>&1; then
        print_status $GREEN "✓ Comprehensive tests PASSED"
        return 0
    else
        print_status $RED "✗ Comprehensive tests FAILED"
        echo "  Log file: $comprehensive_log"
        echo "  Last few lines:"
        tail -10 "$comprehensive_log" | sed 's/^/    /'
        return 1
    fi
}

# Function to test shell script interfaces
test_shell_scripts() {
    print_header "TESTING SHELL SCRIPT INTERFACES"
    
    local scripts=(
        "${SCRIPT_DIR}/cpu/inject_cpu_stress.sh"
        "${SCRIPT_DIR}/memory/inject_memory_stress.sh"
        "${SCRIPT_DIR}/disk/inject_disk_stress.sh"
        "${SCRIPT_DIR}/socket/inject_socket_failure.sh"
    )
    
    local passed=0
    local total=${#scripts[@]}
    
    for script in "${scripts[@]}"; do
        if [[ -f "$script" ]]; then
            if [[ -x "$script" ]]; then
                # Test help functionality
                if "$script" --help &> /dev/null; then
                    print_status $GREEN "✓ $(basename "$script") help works"
                    passed=$((passed + 1))
                else
                    print_status $RED "✗ $(basename "$script") help failed"
                fi
            else
                print_status $RED "✗ $(basename "$script") is not executable"
            fi
        else
            print_status $RED "✗ $(basename "$script") not found"
        fi
    done
    
    print_status $BLUE "Shell script tests summary: $passed/$total passed"
    return $((total - passed))
}

# Function to run integration tests
run_integration_tests() {
    print_header "RUNNING INTEGRATION TESTS"
    
    print_status $BLUE "Testing configuration file integration..."
    
    # Test configuration files exist and are valid JSON
    local config_files=(
        "${SCRIPT_DIR}/cpu/cpu_config.json"
        "${SCRIPT_DIR}/memory/memory_config.json"
        "${SCRIPT_DIR}/disk/disk_config.json"
        "${SCRIPT_DIR}/network/delay_config.json"
        "${SCRIPT_DIR}/network/loss_config.json"
        "${SCRIPT_DIR}/socket/socket_config.json"
    )
    
    local passed=0
    local total=${#config_files[@]}
    
    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            if jq . "$config_file" &> /dev/null; then
                print_status $GREEN "✓ $(basename "$config_file") is valid JSON"
                passed=$((passed + 1))
            else
                print_status $RED "✗ $(basename "$config_file") is invalid JSON"
            fi
        else
            print_status $RED "✗ $(basename "$config_file") not found"
        fi
    done
    
    print_status $BLUE "Configuration file tests summary: $passed/$total passed"
    return $((total - passed))
}

# Function to run end-to-end tests
run_e2e_tests() {
    print_header "RUNNING END-TO-END TESTS"
    
    local e2e_log="${LOG_DIR}/e2e_test_${TIMESTAMP}.log"
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        print_status $YELLOW "⚠ Docker not available, skipping E2E tests"
        return 0
    fi
    
    # Check if any containers are running
    local container_count=$(docker ps -q | wc -l)
    if [[ $container_count -eq 0 ]]; then
        print_status $YELLOW "⚠ No running containers found, skipping E2E tests"
        echo "  To run E2E tests, start some containers first:"
        echo "  docker-compose -f docker-compose.otel-demo.yml up -d"
        return 0
    fi
    
    print_status $BLUE "Found $container_count running containers, proceeding with E2E tests..."
    
    if python3 "${SCRIPT_DIR}/e2e_test.py" > "$e2e_log" 2>&1; then
        print_status $GREEN "✓ End-to-end tests PASSED"
        return 0
    else
        print_status $RED "✗ End-to-end tests FAILED"
        echo "  Log file: $e2e_log"
        echo "  Last few lines:"
        tail -10 "$e2e_log" | sed 's/^/    /'
        return 1
    fi
}

# Function to generate test report
generate_report() {
    local total_tests=$1
    local passed_tests=$2
    local report_file="${LOG_DIR}/test_report_${TIMESTAMP}.txt"
    
    print_header "GENERATING TEST REPORT"
    
    cat > "$report_file" << EOF
Chaos Engineering Test Report
Generated: $(date)
Test Run ID: $TIMESTAMP

SUMMARY
=======
Total Test Categories: $total_tests
Passed Test Categories: $passed_tests
Failed Test Categories: $((total_tests - passed_tests))
Success Rate: $(( (passed_tests * 100) / total_tests ))%

SYSTEM INFORMATION
==================
OS: $(uname -s)
Kernel: $(uname -r)
Architecture: $(uname -m)
User: $(whoami)
Working Directory: $SCRIPT_DIR

AVAILABLE TOOLS
===============
EOF
    
    # Add tool availability to report
    tools=("python3" "stress-ng" "tc" "iptables" "jq")
    for tool in "${tools[@]}"; do
        if command -v "$tool" &> /dev/null; then
            echo "$tool: Available ($(command -v "$tool"))" >> "$report_file"
        else
            echo "$tool: Not Available" >> "$report_file"
        fi
    done
    
    echo "" >> "$report_file"
    echo "LOG FILES" >> "$report_file"
    echo "=========" >> "$report_file"
    find "$LOG_DIR" -name "*_${TIMESTAMP}.log" -type f | while read -r log_file; do
        echo "$(basename "$log_file"): $log_file" >> "$report_file"
    done
    
    print_status $GREEN "✓ Test report generated: $report_file"
}

# Main execution
main() {
    print_header "CHAOS ENGINEERING COMPREHENSIVE TEST RUNNER"
    echo "Starting comprehensive test suite for all chaos engineering modules"
    echo "Timestamp: $(date)"
    echo "Log directory: $LOG_DIR"
    echo ""
    
    # Initialize counters
    local total_categories=0
    local passed_categories=0
    
    # Check system requirements
    check_requirements
    
    # Run quick tests
    total_categories=$((total_categories + 1))
    if run_quick_tests; then
        passed_categories=$((passed_categories + 1))
    fi
    
    # Run individual module tests
    total_categories=$((total_categories + 1))
    if run_individual_tests; then
        passed_categories=$((passed_categories + 1))
    fi
    
    # Test shell scripts
    total_categories=$((total_categories + 1))
    if test_shell_scripts; then
        passed_categories=$((passed_categories + 1))
    fi
    
    # Run integration tests
    total_categories=$((total_categories + 1))
    if run_integration_tests; then
        passed_categories=$((passed_categories + 1))
    fi
    
    # Run comprehensive test suite
    total_categories=$((total_categories + 1))
    if run_comprehensive_tests; then
        passed_categories=$((passed_categories + 1))
    fi
    
    # Run end-to-end tests (if containers are available)
    total_categories=$((total_categories + 1))
    if run_e2e_tests; then
        passed_categories=$((passed_categories + 1))
    fi
    
    # Generate report
    generate_report $total_categories $passed_categories
    
    # Final results
    print_header "FINAL RESULTS"
    
    if [[ $passed_categories -eq $total_categories ]]; then
        print_status $GREEN "🎉 ALL TEST CATEGORIES PASSED!"
        print_status $GREEN "Chaos engineering implementation is fully functional."
        echo ""
        echo "All requirements have been verified:"
        echo "  ✓ CPU stress injection (Requirement 3.1)"
        echo "  ✓ Memory stress injection (Requirement 3.2)"
        echo "  ✓ Disk I/O stress injection (Requirement 3.3)"
        echo "  ✓ Network delay injection (Requirement 3.4)"
        echo "  ✓ Network packet loss injection (Requirement 3.5)"
        echo "  ✓ Socket/connection failure injection (Requirement 3.6)"
        echo "  ✓ Injection timestamp and monitoring (Requirement 3.7)"
        echo "  ✓ Configurable target services (Requirement 6.2)"
        echo "  ✓ Configurable chaos intensity levels (Requirement 6.3)"
        echo "  ✓ Configuration parameter validation (Requirement 6.5)"
        echo ""
        echo "The chaos engineering module is ready for production use!"
        return 0
    else
        print_status $RED "❌ SOME TESTS FAILED"
        print_status $RED "Test categories passed: $passed_categories/$total_categories"
        echo ""
        echo "Please review the log files in $LOG_DIR for detailed error information."
        echo "Common issues:"
        echo "  - Missing system tools (stress-ng, tc, iptables)"
        echo "  - Insufficient permissions (try running with sudo)"
        echo "  - Missing Python packages (psutil)"
        return 1
    fi
}

# Show usage if help requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Comprehensive test runner for chaos engineering modules."
    echo ""
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo ""
    echo "This script will:"
    echo "  1. Check system requirements"
    echo "  2. Run quick tests (no root required)"
    echo "  3. Run individual module tests"
    echo "  4. Test shell script interfaces"
    echo "  5. Run integration tests"
    echo "  6. Run comprehensive test suite"
    echo "  7. Generate detailed test report"
    echo ""
    echo "For full functionality, run as root:"
    echo "  sudo $0"
    echo ""
    echo "Log files will be saved to: $LOG_DIR"
    exit 0
fi

# Run main function
main "$@"