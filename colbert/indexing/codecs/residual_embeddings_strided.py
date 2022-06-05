# from colbert.indexing.codecs.residual import ResidualCodec
import colbert.indexing.codecs.residual_embeddings as residual_embeddings

from colbert.search.strided_tensor import StridedTensor

"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""


class ResidualEmbeddingsStrided:
    def __init__(self, codec, embeddings, doclens, tokens=None):
        self.codec = codec
        self.codes = embeddings.codes
        self.residuals = embeddings.residuals
        self.tokens = tokens

        self.codes_strided = StridedTensor(self.codes, doclens)
        self.residuals_strided = StridedTensor(self.residuals, doclens)
        if self.tokens is not None:
            self.tokens_strided = StridedTensor(self.tokens, doclens)

    def lookup_eids(self, embedding_ids, codes=None, out_device='cuda'):
        codes = self.codes[embedding_ids] if codes is None else codes
        residuals = self.residuals[embedding_ids]

        return self.codec.decompress(residual_embeddings.ResidualEmbeddings(codes, residuals))

    # # @profile
    def lookup_pids(self, passage_ids, out_device='cuda', return_tokens: bool = False):
        codes_packed, codes_lengths = self.codes_strided.lookup(passage_ids)#.as_packed_tensor()
        residuals_packed, _ = self.residuals_strided.lookup(passage_ids)#.as_packed_tensor()
        embeddings_packed = self.codec.decompress(residual_embeddings.ResidualEmbeddings(codes_packed, residuals_packed))
        if return_tokens and hasattr(self, 'tokens_strided'):
            tokens_packed, _ = self.tokens_strided.lookup(passage_ids)
            return embeddings_packed, codes_lengths, tokens_packed
        return embeddings_packed, codes_lengths, None
        # return StridedTensor(embeddings_packed, codes_lengths).as_padded_tensor()
        # return StridedTensor.pad_packed(embeddings_packed, codes_lengths)
