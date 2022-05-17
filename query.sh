#!/usr/bin/env bash

data_root=../FiD/open_domain_data

num_gpu=$1
index_short_name=$2
model_name=$3
other="${@:4}"

if [[ ${model_name} == 'ms' ]]; then
  model=downloads/colbertv2.0
elif [[ ${model_name} == 'nq' ]]; then
  model=downloads/colbert-60000.dnn
fi

if [[ ${index_short_name} == 'nq_test_top10' ]]; then
  queries=${data_root}/NQ/test.json
  passages=${data_root}/NQ/psgs_w100.test_top10_aggregate.tsv
  passage_maxlength=200
elif [[ ${index_short_name} == 'nq' ]]; then
  queries=${data_root}/NQ/test.json
  passages=${data_root}/NQ/psgs_w100.tsv
  passage_maxlength=200
elif [[ ${index_short_name} == 'msmarcoqa_dev' ]]; then
  queries=${data_root}/msmarco_qa/dev.json
  passages=${data_root}/msmarco_qa/psgs.dev_aggregate.tsv
  passage_maxlength=200
elif [[ ${index_short_name} == 'bioasq_500k_test' ]]; then
  queries=${data_root}/bioasq_500k.nosummary/test.json
  passages=${data_root}/bioasq_500k.nosummary/psgs.test_aggregate.tsv
  passage_maxlength=512  # TODO cannot afford 1024
elif [[ ${index_short_name} == 'fiqa' ]]; then
  queries=${data_root}/fiqa/test.json
  passages=${data_root}/fiqa/psgs.tsv
  passage_maxlength=512
elif [[ ${index_short_name} == 'cqadupstack_mathematica' ]]; then
  queries=${data_root}/cqadupstack/mathematica/test.json
  passages=${data_root}/cqadupstack/mathematica/psgs.tsv
  passage_maxlength=512
elif [[ ${index_short_name} == 'cqadupstack_physics' ]]; then
  queries=${data_root}/cqadupstack/physics/test.json
  passages=${data_root}/cqadupstack/physics/psgs.tsv
  passage_maxlength=512
elif [[ ${index_short_name} == 'cqadupstack_programmers' ]]; then
  queries=${data_root}/cqadupstack/programmers/test.json
  passages=${data_root}/cqadupstack/programmers/psgs.tsv
  passage_maxlength=512
else
  exit
fi

python run.py \
  --model ${model} \
  --index_name ${index_short_name} \
  --queries ${queries} \
  --passages ${passages} \
  --passage_maxlength ${passage_maxlength} \
  --ngpu ${num_gpu} \
  --doc_topk 10 \
  ${other}
