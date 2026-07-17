#!/bin/bash

MODE=$1

cd "$(dirname "$0")"

MPI_PROCESSES=3

datasets=(
    abalone
    car-eval-4
    ai4i2020
    airlines
    bank32nh
    bank-customer-churn-prediction
    bank-marketing
    churn
    fried
    japanese-vowels
    lines-overload-50
    magic-gamma-telescope
    mammography
    nhanes
    online-shoppers-purchasing-intention
    pen-digits
    seismic-bumps
    ur3-cobot-ops
    wilt
    rl
)


run_selection() {

   echo "[INFO] Evaluating underlying generators to identify the best candidate for K-IPO"

    base="./selection"

    generators=(
        CTGAN
        GaussianCopula
        KIPO
        SMOTE
        TabDDPM
        TVAE
    )

    for dataset in "${datasets[@]}"; do
        for generator in "${generators[@]}"; do
             for i in {1..10}; do

                csv_file="${base}/${dataset}/datasets/${generator}/${i}.csv"
                
                if [ -f "$csv_file" ]; then
                    used_gen="$generator"
                elif [ "$generator" == "SMOTE" ] && [ -f "${base}/${dataset}/datasets/SMOTENC/${i}.csv" ]; then
                    csv_file="${base}/${dataset}/datasets/SMOTENC/${i}.csv"
                    used_gen="SMOTENC"
                elif [ "$generator" == "SMOTE" ] && [ -f "${base}/${dataset}/datasets/SMOTEN/${i}.csv" ]; then
                    csv_file="${base}/${dataset}/datasets/SMOTEN/${i}.csv"
                    used_gen="SMOTEN"
                else
                    if [ "$generator" == "SMOTE" ]; then
                        echo "[WARNING] Skipping ${dataset} run ${i}: no valid synthetic dataset found for SMOTE variants (SMOTE, SMOTENC, SMOTEN)"
                    else
                        echo "[WARNING] Skipping ${dataset} run ${i}: no valid synthetic dataset found for generator ${generator}"
                    fi
                    continue
                fi
                
                abs_path=$(realpath "${csv_file}")
                echo "[INFO] Dataset = ${dataset} | Generator = ${used_gen} | File = ${abs_path}"

                mpirun \
                    -n "${MPI_PROCESSES}" python3 eval.py --dataset "${dataset}" --eval_dataset "${csv_file}" --no-xai

            done
        done
    done

    echo "[INFO] Performance comparison completed."
}


run_sensitivity_analysis() {

    echo "[INFO] Running sensitivity analysis of K-IPO"

    base="./sensitivity_analysis/datasets"

    for dataset in "${datasets[@]}"; do

        data_dir="${base}/${dataset}"

        if [ ! -d "${data_dir}" ]; then
            echo "[WARNING] Dataset directory not found: ${data_dir}"
            continue
        fi

        csv_files=("${data_dir}"/*.csv)

        if [ ! -e "${csv_files[0]}" ]; then
            echo "[WARNING] No CSV files found for dataset: ${dataset}"
            continue
        fi


        for csv_file in "${csv_files[@]}"; do

            abs_path=$(realpath "${csv_file}")
            echo "[INFO] Dataset = ${dataset} | File = ${abs_path}"

            mpirun \
                -n "${MPI_PROCESSES}" python3 eval.py --dataset "${dataset}" --eval_dataset "${csv_file}"

        done

    done

    echo "[INFO] Sensitivity analysis completed."
}


run_eval() {

   echo "[INFO] Running performance comparison of K-IPO against other data generators"

    base="./evaluation"

    generators=(
        CTGAN
        GaussianCopula
        KIPO
        SMOTE
        TabDDPM
        TVAE
    )

    for dataset in "${datasets[@]}"; do
        for generator in "${generators[@]}"; do
             for i in {1..10}; do

                csv_file="${base}/${dataset}/datasets/${generator}/${i}.csv"
                if [ -f "$csv_file" ]; then
                    used_gen="$generator"
                elif [ -f "${base}/${dataset}/datasets/SMOTENC/${i}.csv" ]; then
                    csv_file="${base}/${dataset}/datasets/SMOTENC/${i}.csv"
                    used_gen="SMOTENC"
                elif [ -f "${base}/${dataset}/datasets/SMOTEN/${i}.csv" ]; then
                    csv_file="${base}/${dataset}/datasets/SMOTEN/${i}.csv"
                    used_gen="SMOTEN"
                else
                    if [ "$generator" == "SMOTE" ]; then
                        echo "[WARNING] Skipping ${dataset} run ${i}: no valid synthetic dataset found for SMOTE variants (SMOTE, SMOTENC, SMOTEN)"
                    else
                        echo "[WARNING] Skipping ${dataset} run ${i}: no valid synthetic dataset found for generator ${generator}"
                    fi
                    continue
                
                fi
                
                abs_path=$(realpath "${csv_file}")
                echo "[INFO] Dataset = ${dataset} | Generator = ${used_gen} | File = ${abs_path}"

                mpirun \
                    -n "${MPI_PROCESSES}" python3 eval.py --dataset "${dataset}" --eval_dataset "${csv_file}"

            done
        done
    done

    echo "[INFO] Performance comparison completed."
}


case "$MODE" in
    A) run_selection ;;
    B) run_sensitivity_analysis ;;
    C) run_eval ;;
    *) 
        echo "Usage: $0 {A|B|C}"
        echo "A: Evaluating underlying generators to identify the best candidate for K-IPO"
        echo "B: K-IPO sensitivity analysis"
        echo "C: Performance comparison of K-IPO with other data generators"
        exit 1
        ;;
esac