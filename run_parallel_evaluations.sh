#!/bin/bash

# Script to run TATQA, FinQA, and FETAQA evaluations in parallel
# Rate limits: 60 requests/minute = 1 request/second
# Using 4-second delays between requests for safety
# Each script has retry logic to handle rate limit errors gracefully

echo "🚀 Starting parallel evaluation of TATQA, FinQA, and FETAQA datasets"
echo "📊 Rate limits: 60 requests/minute (1 request/second)"
echo "⏱️  Using 4-second delays between requests for safety"
echo "🔄 Each script has retry logic to handle rate limit errors"
echo ""

# Function to run evaluation in background
run_evaluation() {
    local dataset=$1
    local script_name=$2
    local comment=$3
    
    echo "Starting $dataset evaluation in background..."
    echo "Command: ./$script_name --provider mistral --model mistral-small-latest --method ee_fs --comment $comment"
    echo ""
    
    ./$script_name --provider mistral --model mistral-small-latest --method ee_fs --comment $comment &
    local pid=$!
    echo "✅ $dataset evaluation started with PID: $pid"
    echo ""
}

# Get timestamp for unique comment
timestamp=$(date +"%Y%m%d_%H%M%S")
comment="parallel_run_${timestamp}"

echo "🕐 Starting all evaluations at: $(date)"
echo "📝 Comment: $comment"
echo ""

# Start all three evaluations in parallel
run_evaluation "TATQA" "run_tatqa.sh" "$comment"
run_evaluation "FinQA" "run_finqa.sh" "$comment" 
run_evaluation "FETAQA" "run_fetaqa.sh" "$comment"

echo "🎯 All three evaluations are now running in parallel!"
echo "📊 Monitor progress with: ps aux | grep run_"
echo "📁 Check output directories for results"
echo "⏹️  To stop all: pkill -f 'run_amaf.py'"
echo ""
echo "Waiting for all processes to complete..."
echo ""

# Wait for all background processes to complete
wait

echo ""
echo "🎉 All parallel evaluations completed!"
echo "📅 Finished at: $(date)" 