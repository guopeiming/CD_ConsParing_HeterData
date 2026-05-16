# ！/bin/bash
set -e

export TOKENIZERS_PARALLELISM="false"

data_dir="data/base/consti_parsing"
model_dir="results/joint_lm_cp/all_1w/"
res_dir=$model_dir/res
if [ ! -d $res_dir ]; then
    mkdir $res_dir
fi

allennlp evaluate \
    $model_dir \
    $data_dir/dialogue.jsonl,$data_dir/forum.jsonl,$data_dir/law.jsonl,$data_dir/literature.jsonl,$data_dir/review.jsonl \
    --output-file $res_dir/dialogue.jsonl,$res_dir/forum.jsonl,$res_dir/law.jsonl,$res_dir/literature.jsonl,$res_dir/review.jsonl \
    --cuda-device 0 \
    -o "{\"validation_data_loader.batch_sampler.batch_size\": 30}"
