datasets=(coco aokvqa gqa)
types=(random popular adversarial)

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do

        python ./eval/pope_eval_base.py \
            --ref-files ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --res-files ./outputs/pope/baseline/llava-7b-${dataset}-${type}-greedy.jsonl
    done
done