from typing import List, Dict, Union
import os
import sys
import json
import argparse
import torch
import numpy as np
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

try:
  torch.inference_mode
except:
  print('use torch.no_grad as torch.inference_mode')
  torch.inference_mode = torch.no_grad

dataroot = 'downloads/lotte'
dataset = 'lifestyle'
datasplit = 'dev'
queries = os.path.join(dataroot, dataset, datasplit, 'questions.search.tsv')
passages = os.path.join(dataroot, dataset, datasplit, 'collection.tsv')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='colbert')
  parser.add_argument('--exp', type=str, help='name of the experiment', default=None)
  parser.add_argument('--model', type=str, help='model checkpoints', default='downloads/colbertv2.0')
  parser.add_argument('--fid', type=str, help='fid model checkpoints', default=None)
  parser.add_argument('--fid_head_index', type=int, help='fid head index to use', default=None)
  parser.add_argument('--index_name', type=str, help='index name', default=f'{dataset}.{datasplit}')
  parser.add_argument('--queries', type=str, help='query file', default=queries)
  parser.add_argument('--passages', type=str, help='passage file', default=passages)
  parser.add_argument('--output', type=str, help='output file', default=None)
  parser.add_argument('--passage_maxlength', type=int, default=300, help='maximum number of tokens in a passage')
  parser.add_argument('--nbits', type=int, help='bits of each dimension', default=2)
  parser.add_argument('--kmeans_niters', type=int, help='number of iterations in kmeans', default=20)
  parser.add_argument('--num_partitions', type=str, help='num of clusters in index', default='default',
                      choices=['default', 'divide10', 'const10', 'const10000'])
  parser.add_argument('--doc_topk', type=int, help='topk documents returned', default=10)
  parser.add_argument('--ngpu', type=int, help='num of gpus', default=1)
  parser.add_argument('--nprobe', type=int, help='num of cluster to query', default=2)
  parser.add_argument('--ncandidates', type=int, help='num of doc to rerank', default=8192)
  parser.add_argument('--no_rerank', action='store_true', help='no reranking')
  parser.add_argument('--use_real_tokens', action='store_true', help='only use real tokens without pad or mask tokens')
  parser.add_argument('--no_half', action='store_true', help='disable half precision')
  parser.add_argument('--no_norm', action='store_true', help='disable normalization')
  parser.add_argument('--overwrite', type=str, help='whether overwrite the index', default='reuse')
  parser.add_argument('--onlyid', action='store_true', help='only store id in the final file to save memory')
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  use_fid = args.fid is not None
  use_fid_format = use_fid
  if use_fid:
    args.exp = args.fid[args.fid.find('trained_reader'):].rsplit('/')[1]
    extra = {'dim': 64}
  else:
    args.exp = args.model.rsplit('/', 1)[1] if args.exp is None else args.exp
    extra = {}
  if args.debug:
    args.overwrite = True
    args.exp = 'debug'
  index_name = f'{args.index_name}.{args.nbits}bits' \
               f'{("." + str(args.kmeans_niters) + "iter") if args.kmeans_niters != 20 else ""}' \
               f'{("." + args.num_partitions + "partition") if args.num_partitions != "default" else ""}' \
               f'{".nohalf" if args.no_half else ""}'

  # load data
  queries = Queries(path=args.queries, use_fid_format=use_fid_format)
  collection = Collection(path=args.passages, use_fid_format=use_fid_format, keep_raw=not args.onlyid)
  if args.debug:  # use the first 1k to debug
    collection.keep(5000)
  print(f'Loaded {len(queries)} queries and {len(collection)} passages')
  print(f'QUERY: {queries[np.random.choice(len(queries))]}')
  print(f'DOC: {collection[np.random.choice(len(collection))]}')

  # index
  with Run().context(RunConfig(nranks=args.ngpu, experiment=args.exp)):
    config = ColBERTConfig(doc_maxlen=args.passage_maxlength,
                           nbits=args.nbits,
                           num_partitions=args.num_partitions,
                           fid_model_path=args.fid,
                           fid_head_index=args.fid_head_index,
                           half_precision=not args.no_half,
                           normalize=not args.no_norm,
                           kmeans_niters=args.kmeans_niters,
                           **extra)
    indexer = Indexer(checkpoint=args.model, config=config)
    indexer.index(name=index_name, collection=collection, overwrite=args.overwrite)

  # query
  with Run().context(RunConfig(experiment=args.exp)):
    searcher = Searcher(index=index_name)
    searcher.configure(nprobe=args.nprobe, ncandidates=args.ncandidates, no_rerank=args.no_rerank, use_real_tokens=args.use_real_tokens)

  # test
  query = queries[0]
  print(f'#> {query}')
  results = searcher.search(query, k=3)
  for id, rank, score in zip(*results):
    print(f'\t [{rank}] \t\t {score:.1f} \t\t {searcher.collection[id]}')

  # output
  args.output = os.path.join(
    'experiments', args.exp, 'indexes', index_name,
    f'result_{args.nprobe}probe' + 
    f'_{args.ncandidates}cand' + 
    ('_norerank' if args.no_rerank else '') + 
    ('_real' if args.use_real_tokens else '') + 
    ('_onlyid' if args.onlyid else '') + 
    '.json') \
    if args.output is None else args.output
  rankings = searcher.search_all(queries, k=args.doc_topk).todict()
  for i in range(len(rankings)):
    ctxs: List[Dict] = [{**collection.id2raw[id], **{'score': score}} for id, rank, score in rankings[i]]
    queries.raw_queries[i]['ctxs'] = ctxs
  with open(args.output, 'w') as fout:
    json.dump(queries.raw_queries, fout, indent=2)
