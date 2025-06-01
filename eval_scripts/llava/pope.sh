seed=${1:-55}
dataset_name=${2:-"coco"}
type=${3:-"popular"}
model_path=${4:-"/data/hall_mllms/CausalMM/llava-v1.5-7b"}
gamma=${5:-1.0}
epsilon=${6:-0.6}
if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=/data/hall_mllms/CausalMM/coco/val2014
else
  image_folder=/data/hall_mllms/CausalMM/gqa/images
fi

python -m llava.eval.llava_model_vqa_loader_RL \
--model-path ${model_path} \
--question-file /data/hall_mllms/CausalMM/llava-1.5/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ./playground/data/eval/pope/output/llava15_${dataset_name}_pope_${type}_answers.jsonl \
--temperature 0 \
--cuda-device 'cuda:0' \
--apply-memvr 'memvr' \
--retracing-ratio 0.12 \
--entropy-threshold 0.75 \
--max-new-tokens 5 \
--starting-layer 5 \
--ending-layer 16 \

python /data/hall_mllms/CausalMM/llava-1.5/experiments/eval/eval_pope.py \
--gt_files /data/hall_mllms/CausalMM/llava-1.5/experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--gen_files /data/hall_mllms/LLaVA/playground/data/eval/pope/output/llava15_${dataset_name}_pope_${type}_answers.jsonl 


