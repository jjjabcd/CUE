#!/bin/bash

metrics=("DIS" "INH")
methods=("PI" "Ssel" "WS2" "RS2")

# 제공해주신 하위 설정(Configuration) 목록을 배열로 등록합니다.
configs=(
    "CONCAT"
    "CUSTOM_FP25_IMG75"
    "CUSTOM_FP75_IMG25"
    "LTAU_BIN10000_K3"
    "LTAU_BIN10000_K5"
    "LTAU_BIN10000_K7"
    "LTAU_BIN1250_K3"
    "LTAU_BIN1250_K5"
    "LTAU_BIN1250_K7"
    "LTAU_BIN2500_K3"
    "LTAU_BIN2500_K5"
    "LTAU_BIN2500_K7"
    "LTAU_BIN5000_K3"
    "LTAU_BIN5000_K5"
    "LTAU_BIN5000_K7"
    "TAU0.25"
    "TAU0.5"
    "TAU1.0"
    "TAU2.0"
    "TAU4.0"
    "UNIFORM"
)

for metric in "${metrics[@]}"; do
    for method in "${methods[@]}"; do
        for config in "${configs[@]}"; do
            
            # 대상 디렉토리 경로 지정 (R{num} 대신 config 변수 사용)
            target_dir="results/${metric}/${method}/v1/${config}/"
            
            # 해당 디렉토리가 실제로 존재하는지 확인
            if [ -d "$target_dir" ]; then
                
                # 1. 해당 디렉토리만 add
                git add "$target_dir"
                
                # 2. 스테이징 영역에 추가된 변경사항이 있는지 확인
                if ! git diff --cached --quiet; then
                    # 커밋 메시지 형식 수정
                    commit_msg="Upload ${metric}/${method}/v1/${config}"
                    
                    git commit -m "$commit_msg"
                    git push origin main
                else
                    echo "✨ No changes detected in $target_dir"
                fi
            fi
            
        done
    done
done

echo "========================================"
echo "✅ All processes completed!"