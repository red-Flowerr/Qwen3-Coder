set -x
mkdir -p results/humaneval
export HF_ENDPOINT=https://hf-mirror.com
export PATH=./vllm/bin:$PATH
export PYTHONPATH=$PYTHONPATH:./eval_plus/evalplus

# pip install datamodel_code_generator anthropic mistralai google-generativeai
MODEL_DIR="/mnt/hdfs/tiktok_aiic_new/user/longtao.zheng/verl_rl_checkpoints_debug/verl-sft-lr1e-4_OpenThoughts3_Qwen3-4B-Base/global_step_2343"
MODEL_NAME="qwen3-4b-sft-lr-4"
TP=2
OUTPUT_DIR="./results"
TEMPERATURE=0.6
export CUDA_VISIBLE_DEVICES=0,1

mkdir -p ${OUTPUT_DIR}

echo "EvalPlus: ${MODEL_DIR}, OUTPUT_DIR ${OUTPUT_DIR}"

python generate.py \
  --model_type qwen2 \
  --model_size chat \
  --model_path ${MODEL_DIR} \
  --bs 1 \
  --temperature ${TEMPERATURE} \
  --n_samples 1 \
  --root ${OUTPUT_DIR} \
  --dataset humaneval \
  --tensor-parallel-size ${TP} \
  --model_name ${MODEL_NAME}


python -m evalplus.sanitize --samples "${OUTPUT_DIR}/mbpp/qwen2_chat_temp_${TEMPERATURE}_${MODEL_NAME}"

evalplus.evaluate \
  --dataset humaneval \
  --samples "${OUTPUT_DIR}/mbpp/qwen2_chat_temp_${TEMPERATURE}_${MODEL_NAME}" > "${OUTPUT_DIR}/raw_humaneval_${MODEL_NAME}_results.txt"

evalplus.evaluate \
  --dataset humaneval \
  --samples "${OUTPUT_DIR}/mbpp/qwen2_chat_temp_${TEMPERATURE}_${MODEL_NAME}-sanitized" > "${OUTPUT_DIR}/humaneval_${MODEL_NAME}_results.txt"


python generate.py \
  --model_type qwen2 \
  --model_size chat \
  --model_path ${MODEL_DIR} \
  --bs 1 \
  --temperature ${TEMPERATURE} \
  --n_samples 1 \
  --root ${OUTPUT_DIR} \
  --dataset mbpp \
  --tensor-parallel-size ${TP} \
  --model_name ${MODEL_NAME}


python -m evalplus.sanitize --samples "${OUTPUT_DIR}/mbpp/qwen2_chat_temp_${TEMPERATURE}_${MODEL_NAME}"


evalplus.evaluate \
  --dataset mbpp \
  --samples "${OUTPUT_DIR}/mbpp/qwen2_chat_temp_${TEMPERATURE}_${MODEL_NAME}" > "${OUTPUT_DIR}/raw_mbpp_${MODEL_NAME}_results.txt"

evalplus.evaluate \
  --dataset mbpp \
  --samples "${OUTPUT_DIR}/mbpp/qwen2_chat_temp_${TEMPERATURE}_${MODEL_NAME}-sanitized" > "${OUTPUT_DIR}/mbpp_${MODEL_NAME}_results.txt"