import torch

from colbert.utils.utils import flatten, print_message

from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings_strided import ResidualEmbeddingsStrided

from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration

from .index_loader import IndexLoader
from colbert.modeling.colbert import colbert_score, colbert_score_packed


class IndexScorer(IndexLoader, CandidateGeneration):
    def __init__(self, index_path):
        super().__init__(index_path)

        self.embeddings_strided = ResidualEmbeddingsStrided(self.codec, self.embeddings, self.doclens)

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

    def lookup_pids(self, passage_ids, out_device='cuda', return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def retrieve(self, config, Q):
        Q = Q[:, :config.query_maxlen]   # NOTE: Candidate generation uses only the query tokens
        pids, scores = self.generate_candidates(config, Q)

        return pids, scores

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(self.emb2pid[embedding_ids.long()].cuda(), sorted=False)
        return all_pids

    def rank(self, config, Q, input_ids, attention_mask, k):
        with torch.inference_mode():
            if config.use_real_tokens:
                assert Q.size(0) == attention_mask.size(0) == 1
                Q = Q[:, :attention_mask[0].sum()]

            pids, scores = self.retrieve(config, Q)

            if config.no_rerank:
                pids, scores = pids.tolist(), scores.tolist()
            else:
                scores = self.score_pids(config, Q, pids, k)
                scores_sorter = scores.sort(descending=True)
                pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

            return pids, scores

    def score_pids(self, config, Q, pids, k):
        """
            Always supply a flat list or tensor for `pids`.

            Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
            If Q.size(0) is 1, the matrix will be compared with all passages.
            Otherwise, each query matrix will be compared against the *aligned* passage.
        """
        D_packed, D_mask = self.lookup_pids(pids)

        if Q.size(0) == 1:
            return colbert_score_packed(Q, D_packed, D_mask, config)

        D_padded, D_lengths = StridedTensor(D_packed, D_mask).as_padded_tensor()

        return colbert_score(Q, D_padded, D_lengths, config)
