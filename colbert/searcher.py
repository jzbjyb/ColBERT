from locale import normalize
import os
import torch

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.fid_wrapper import FiDCheckpoint
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        self.index = os.path.join(default_index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        if self.config.fid_model_path is not None:
            self.checkpoint = FiDCheckpoint(
                self.config.fid_model_path,
                doc_maxlen=self.config.doc_maxlen,
                query_maxlen=self.config.query_maxlen,
                head_idx=self.config.fid_head_index,
                half_precision=self.config.half_precision,
                normalize=self.config.normalize
            ).cuda()
        else:
            self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config).cuda()
        self.ranker = IndexScorer(self.index)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q, input_ids, attention_mask = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, return_more=True)

        return Q, input_ids, attention_mask

    def search(self, text: str, k=10):
        return self.dense_search(*self.encode(text), k)

    def search_all(self, queries: TextQueries, k=10):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q, input_ids, attention_mask = self.encode(queries_)

        return self._search_all_Q(queries, Q, input_ids, attention_mask, k)

    def _search_all_Q(self, queries, Q, input_ids, attention_mask, k):
        all_scored_pids = [list(zip(
            *self.dense_search(
                Q[query_idx:query_idx+1], 
                input_ids[query_idx:query_idx+1], 
                attention_mask[query_idx:query_idx+1], k=k)))
                for query_idx in tqdm(range(Q.size(0)))]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, k=10):
        qry_tokenizer, doc_tokenizer = self.checkpoint.get_tokenizer('query'), self.checkpoint.get_tokenizer('doc')
        pids, scores, qa_token_pairs = self.ranker.rank(self.config, Q, input_ids, attention_mask, k=k, return_tokens=True)
        ranks = list(range(1, k+1))
        # convert token id to str
        for i, _qa_token_pairs in enumerate(qa_token_pairs):
            qa_token_pairs[i] = [(qry_tokenizer.convert_ids_to_tokens([qt])[0], doc_tokenizer.convert_ids_to_tokens([dt])[0], s) for (qt, dt), s in _qa_token_pairs] 
        return pids, ranks, scores, qa_token_pairs
