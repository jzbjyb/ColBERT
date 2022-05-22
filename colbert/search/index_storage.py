import time
import torch

from colbert.utils.utils import flatten, print_message

from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided

from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration

from colbert.search.index_loader import IndexLoader
from colbert.modeling.colbert import colbert_score, colbert_score_packed


class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, index_path):
        super().__init__(index_path)

        self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens, self.tokens)

        print_message("#> Building the emb2pid mapping..")
        all_doclens = load_doclens(index_path, flatten=False)

        assert self.num_embeddings == sum(flatten(all_doclens))

        all_doclens = flatten(all_doclens)
        total_num_embeddings = sum(all_doclens)

        self.emb2pid = torch.zeros(total_num_embeddings, dtype=torch.int)

        """
        EVENTUALLY: Build this in advance and load it from disk.

        EVENTUALLY: Use two tensors. emb2pid_offsets will have every 256th element. emb2pid_delta will have the delta
                    from the corresponding offset, 
        """

        offset_doclens = 0
        for pid, dlength in enumerate(all_doclens):
            self.emb2pid[offset_doclens: offset_doclens + dlength] = pid
            offset_doclens += dlength

        print_message("len(self.emb2pid) =", len(self.emb2pid))

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        return self.embeddings_strided.lookup_eids(embedding_ids, codes=codes, out_device=out_device)

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False, return_tokens: bool = False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device, return_tokens=return_tokens)

    def retrieve(self, config, Q):
        Q = Q[:, :config.query_maxlen]   # NOTE: Candidate generation uses only the query tokens
        pids, scores = self.generate_candidates(config, Q)

        return pids, scores

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def rank(self, config, Q, query_tokens, query_attn_mask, k, return_tokens: bool = False):
        with torch.inference_mode():
            if config.use_real_tokens:
                assert Q.size(0) == query_attn_mask.size(0) == 1
                Q = Q[:, :query_attn_mask[0].sum()]  # (1, query_len, emb_size)
                query_tokens = query_tokens[:, :query_attn_mask[0].sum()]  # (1, query_len)

            pids, scores = self.retrieve(config, Q)

            qd_token_pairs = [[] for _ in range(len(pids))]
            if config.no_rerank:
                pids, scores = pids[:k].tolist(), scores[:k].tolist()
            else:
                # (num_docs,), (num_docs, query_len), (num_docs, query_len)
                scores, tok_scores, doc_tokens = self.score_pids(config, Q, pids, return_tokens=return_tokens)
                scores, sort_indices = scores.sort(descending=True)
                sort_indices, scores = sort_indices[:k], scores[:k].tolist()
                pids = pids[sort_indices].tolist()
                tok_scores = tok_scores[sort_indices].tolist()
                if return_tokens:
                    # associate qry token with doc token
                    if query_tokens.size(0) == 1:  # (1, query_len) the same query for all docs
                        query_tokens = query_tokens.repeat(doc_tokens.size(0), 1)
                        qd_token_pairs = torch.stack([query_tokens, doc_tokens.to(query_tokens)], dim=-1)
                    elif query_tokens.size(0) == doc_tokens.size(0):  # (num_docs, query_len) different queries for different docs
                        qd_token_pairs = torch.stack([query_tokens, doc_tokens.to(query_tokens)], dim=-1)
                    else:
                        raise ValueError(f'query tokens {query_tokens.size()} and doc tokens {doc_tokens.size()} shape mismatch')
                    qd_token_pairs = qd_token_pairs[sort_indices].tolist()
                    qd_token_pairs = [list(zip(qd_token_pairs[i], tok_scores[i])) for i in range(len(qd_token_pairs))]

            return pids, scores, qd_token_pairs

    def score_pids(self, config, Q, pids, return_tokens: bool = False):
        """
            Always supply a flat list or tensor for `pids`.

            Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
            If Q.size(0) is 1, the matrix will be compared with all passages.
            Otherwise, each query matrix will be compared against the *aligned* passage.
        """
        D_packed, D_mask, tokens_packed = self.lookup_pids(pids, return_tokens=return_tokens)
        if Q.size(0) == 1:
            return colbert_score_packed(Q, D_packed, D_mask, config, tokens_packed=tokens_packed)

        if return_tokens:
            raise NotImplementedError
        
        D_padded, D_lengths = StridedTensor(D_packed, D_mask).as_padded_tensor()
        return colbert_score(Q, D_padded, D_lengths, config)
