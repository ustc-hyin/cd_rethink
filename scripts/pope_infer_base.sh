datasets=(coco aokvqa gqa)
types=(random popular adversarial)

## baseline
for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python ./inference/pope_infer_base.py \
            --model-path /code/pretrained_models/llava-v1.5-7b \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images \
            --answers-file ./outputs/pope/baseline/llava-7b-${dataset}-${type}-greedy.jsonl \
            --temperature 0 \
            --conv-mode vicuna_v1
    done
done

for dataset in ${datasets[@]}; do 
    for type in ${types[@]}; do
        python ./inference/pope_infer_base.py \
            --model-path /code/pretrained_models/llava-v1.5-7b \
            --question-file ./data/pope/${dataset}/${dataset}_pope_${type}.json \
            --image-folder ./data/pope/${dataset}/images \
            --answers-file ./outputs/pope/baseline/llava-7b-${dataset}-${type}-sample.jsonl \
            --temperature 1 \
            --conv-mode vicuna_v1
    done
done