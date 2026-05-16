#!/bin/bash
set -e

# sleep 2h

export TOKENIZERS_PARALLELISM="false"

export seed=0

# export train_data_path='data/base/consti_parsing/train.jsonl'
# export dev_data_path='data/base/consti_parsing/dev.jsonl'
# export test_data_path='data/base/consti_parsing/test.jsonl'
# python train.py -c 1 -n base/base_v2_5e_2 --config BaseConstiParserV2

# export train_data_path='data/joint_lm_cp/train_review_10000.jsonl'
# export dev_data_path='data/base/consti_parsing/dev.jsonl'
# export test_data_path='data/base/consti_parsing/review.jsonl'
# python train.py -c 0 -n joint_lm_cp/review_1w --config JointLMCPParser

# export train_data_path='data/joint_lm_cp/train_all_10000.jsonl'
# export dev_data_path='data/base/consti_parsing/dev.jsonl'
# export test_data_path='data/base/consti_parsing/dialogue.jsonl'
# python train.py -c 1 -n joint_lm_cp/all_1w --config JointLMCPParser_all

# export train_data_path='data/joint_lm_cp_ner/train2.jsonl'
# export dev_data_path='data/base/consti_parsing/dev.jsonl'
# export test_data_path='data/base/consti_parsing/dialogue.jsonl'
# python train.py -c 3 -n joint_lm_cp_ner/train2_new --config JointLMCPNERParser

# export train_data_path='data/joint_lm_cp_ccg/train2.jsonl'
# export dev_data_path='data/base/consti_parsing/dev.jsonl'
# export test_data_path='data/base/consti_parsing/dialogue.jsonl'
# python train.py -c 1 -n joint_lm_cp_ccg/train2 --config JointLMCPCCGParser

# export train_data_path='data/save/train4.jsonl'
# export train_data_path='data/joint_lm_cp_ner_ccg/train5_domain.jsonl'
# export dev_data_path='data/base/consti_parsing/dev.jsonl'
# export test_data_path='data/base/consti_parsing/dialogue.jsonl'
# python train.py -c 1 -n joint_lm_cp_ner_ccg/train5_domain_qkv --config JointLMCPNERCCGParser


export train_data_path='data/base/consti_parsing/fewshot_dialogue_20_1.jsonl'
export dev_data_path='data/base/consti_parsing/fewshot_dialogue_20_1.jsonl'
export test_data_path='data/base/consti_parsing/dialogue.jsonl'
python train.py -c 0 -n fewshot/dia_20_1 --config FewshotBaseConstiParser
