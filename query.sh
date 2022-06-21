#!/usr/bin/env bash

data_root=../FiD/open_domain_data

num_gpu=$1
index_short_name=$2
model_name=$3
other="${@:4}"
save_batch_size=0

if [[ ${model_name} == 'ms' ]]; then
  model=downloads/colbertv2.0
elif [[ ${model_name} == 'nq' ]]; then
  model=downloads/colbert-60000.dnn
fi

if [[ ${index_short_name} == 'nq_test_top10' ]]; then
  queries=${data_root}/NQ/test.json
  passages=${data_root}/NQ/psgs_w100.test_top10_aggregate.tsv
  passage_maxlength=200

elif [[ ${index_short_name} == 'nq_test' ]]; then
  queries=${data_root}/NQ/test.json
  passages=${data_root}/NQ/psgs_w100.test_aggregate.tsv
  passage_maxlength=200

elif [[ ${index_short_name} == 'nq_test_005' ]]; then
  queries=${data_root}/NQ/test.json
  passages=${data_root}/NQ/psgs_w100.test_aggregate_and_005.tsv
  passage_maxlength=200

elif [[ ${index_short_name} == 'nq' ]]; then
  #queries=${data_root}/NQ/test.json
  queries=${data_root}/NQ/train.json
  passages=${data_root}/NQ/psgs_w100.tsv
  passage_maxlength=200
  save_batch_size=5000  # use for training set

elif [[ ${index_short_name} == 'msmarcoqa_dev' ]]; then
  queries=${data_root}/msmarco_qa/dev.json
  passages=${data_root}/msmarco_qa/psgs.dev_aggregate.tsv
  passage_maxlength=200

elif [[ ${index_short_name} == 'msmarco' ]]; then
  queries=${data_root}/msmarco/dev.json
  passages=${data_root}/msmarco/psgs.tsv
  passage_maxlength=200

elif [[ ${index_short_name} == 'bioasq_500k_test' ]]; then
  queries=${data_root}/bioasq_500k.nosummary/test.json
  passages=${data_root}/bioasq_500k.nosummary/psgs.test_aggregate.tsv
  passage_maxlength=512  # TODO cannot afford 1024

elif [[ ${index_short_name} == 'bioasq_500k' ]]; then
  queries=${data_root}/bioasq_500k.nosummary/test.json
  passages=${data_root}/bioasq_500k.nosummary/psgs.tsv
  passage_maxlength=512  # TODO cannot afford 1024

elif [[ ${index_short_name} == 'bioasq_1m' ]]; then
  queries=${data_root}/bioasq_1m/test.json
  passages=${data_root}/bioasq_1m/psgs.tsv
  passage_maxlength=512  # TODO cannot afford 1024

elif [[ ${index_short_name} == 'fiqa' ]]; then
  queries=${data_root}/fiqa/test.json
  passages=${data_root}/fiqa/psgs.tsv
  passage_maxlength=512

elif [[ ${index_short_name} == 'scifact' ]]; then
  queries=${data_root}/scifact/test.json
  passages=${data_root}/scifact/psgs.tsv
  passage_maxlength=512

elif [[ ${index_short_name} == 'nfcorpus' ]]; then
  queries=${data_root}/nfcorpus/test.json
  passages=${data_root}/nfcorpus/psgs.tsv
  passage_maxlength=512

elif [[ ${index_short_name} == 'scidocs' ]]; then
  queries=${data_root}/scidocs/test.json
  passages=${data_root}/scidocs/psgs.tsv
  passage_maxlength=512

elif [[ ${index_short_name} == 'trec_covid' ]]; then
  queries=${data_root}/trec_covid/test.json
  passages=${data_root}/trec_covid/psgs.tsv
  passage_maxlength=512

elif [[ ${index_short_name} == 'touche2020' ]]; then
  queries=${data_root}/touche2020/test.json
  passages=${data_root}/touche2020/psgs.tsv
  passage_maxlength=512

elif [[ ${index_short_name} == 'quora' ]]; then
  queries=${data_root}/quora/test.json
  passages=${data_root}/quora/psgs.tsv
  passage_maxlength=200

elif [[ ${index_short_name} == 'cqadupstack_android' ]]; then
  passages=${data_root}/cqadupstack/android/psgs.tsv
  queries=${data_root}/cqadupstack/android/test.json
  passage_maxlength=512

elif [[ ${index_short_name} == 'cqadupstack_english' ]]; then
  passages=${data_root}/cqadupstack/english/psgs.tsv
  queries=${data_root}/cqadupstack/english/test.json
  passage_maxlength=512

elif [[ ${index_short_name} == 'cqadupstack_gaming' ]]; then
  passages=${data_root}/cqadupstack/gaming/psgs.tsv
  queries=${data_root}/cqadupstack/gaming/test.json
  passage_maxlength=512

elif [[ ${index_short_name} == 'cqadupstack_gis' ]]; then
  passages=${data_root}/cqadupstack/gis/psgs.tsv
  queries=${data_root}/cqadupstack/gis/test.json
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

elif [[ ${index_short_name} == 'cqadupstack_stats' ]]; then
  passages=${data_root}/cqadupstack/stats/psgs.tsv
  queries=${data_root}/cqadupstack/stats/test.json
  passage_maxlength=512

elif [[ ${index_short_name} == 'cqadupstack_tex' ]]; then
  passages=${data_root}/cqadupstack/tex/psgs.tsv
  queries=${data_root}/cqadupstack/tex/test.json
  passage_maxlength=512

elif [[ ${index_short_name} == 'cqadupstack_unix' ]]; then
  passages=${data_root}/cqadupstack/unix/psgs.tsv
  queries=${data_root}/cqadupstack/unix/test.json
  passage_maxlength=512

elif [[ ${index_short_name} == 'cqadupstack_webmasters' ]]; then
  passages=${data_root}/cqadupstack/webmasters/psgs.tsv
  queries=${data_root}/cqadupstack/webmasters/test.json
  passage_maxlength=512

elif [[ ${index_short_name} == 'cqadupstack_wordpress' ]]; then
  passages=${data_root}/cqadupstack/wordpress/psgs.tsv
  queries=${data_root}/cqadupstack/wordpress/test.json
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
  --doc_topk 100 \
  --save_batch_size ${save_batch_size} \
  ${other}
