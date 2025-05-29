#!/bin/bash
# run_energy_calculation.sh

TOTAL=18250
NODES=3
CHUNK=$((TOTAL / NODES))

for ((i=0; i<NODES; i++)); do
    START=$((i * CHUNK))
    END=$(((i+1) * CHUNK - 1))
    # 마지막 노드는 나머지 모두 처리
    if [ $i -eq $((NODES-1)) ]; then
        END=$TOTAL
    fi
    
    OUTPUT="results_node${i}.csv"
    
    # 백그라운드로 실행
    nohup python energy_calculator.py your_data.csv --cpu 8 --start $START --end $END --output $OUTPUT > log_node${i}.txt 2>&1 &
    
    echo "Started node $i (PID: $!) processing range $START-$END"
done
