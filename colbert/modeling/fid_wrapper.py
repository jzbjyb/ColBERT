import torch
from transformers import T5Tokenizer
import sys
sys.path.insert(0, '/home/jzh1syv/exp/FiD')
from src.model import FiDT5
from colbert.infra.run import Run
from colbert.utils.utils import batch

class FiDCheckpoint():
    def __init__(self,
                 model_path: str,
                 doc_maxlen: int,
                 query_maxlen: int,
                 head_idx: int,
                 bsize: int = None,
                 half_precision: bool = True):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        self.model = FiDT5.from_pretrained(model_path)
        self.model.eval()
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.head_idx = head_idx
        self.bsize = bsize
        self.half_precision = half_precision
        self.normalize = True  # consistent with colbert
        self.query_tokenizer = type('placeholder', (object,), {'query_maxlen': None})()

    def cuda(self):
        self.model.cuda()
        return self

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None):
        if context is not None:
            raise NotImplementedError
        if bsize is None:
            bsize = len(queries)
        with torch.no_grad():
            embs, masks = [], []
            for batch_ind in range(0, len(queries), bsize):
                batch = queries[batch_ind:batch_ind + bsize]
                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch,
                    padding='max_length',
                    return_tensors='pt',
                    max_length=self.query_maxlen,
                    truncation=True)
                ids = encoded_batch['input_ids'].to(self.model.device)
                mask = encoded_batch['attention_mask'].bool().to(self.model.device)
                emb = self.model.encode_query(
                    input_ids=ids,
                    attention_mask=mask)[:, 0, self.head_idx]  # layer 0
                emb = emb * mask.to(emb).unsqueeze(-1)
                if self.normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=2)
                if self.half_precision:
                    emb = emb.half()
                embs.append(emb)
                masks.append(mask)
            embs = torch.cat(embs)
            if to_cpu:
                embs = embs.cpu()
            return embs

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        if keep_dims != 'flatten' or to_cpu != False or showprogress != False or return_tokens != False:
            raise NotImplementedError
        if bsize is None:
            bsize = len(docs)
        with torch.no_grad():
            embs, masks = [], []
            for batch_ind in range(0, len(docs), bsize):
                batch = docs[batch_ind:batch_ind + bsize]
                encoded_batch = self.tokenizer.batch_encode_plus(
                    batch,
                    padding='max_length',
                    return_tensors='pt',
                    max_length=self.doc_maxlen,
                    truncation=True)
                ids = encoded_batch['input_ids'].to(self.model.device)
                mask = encoded_batch['attention_mask'].bool().to(self.model.device)
                emb = self.model.encode_context(
                    input_ids=ids,
                    attention_mask=mask)[:, 0, self.head_idx]  # layer 0
                emb = emb * mask.to(emb).unsqueeze(-1)
                if self.normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=2)
                if self.half_precision:
                    emb = emb.half()
                embs.append(emb)
                masks.append(mask)
            embs, masks = torch.cat(embs), torch.cat(masks)
            doclens = masks.sum(-1).tolist()
            embs = embs.view(-1, embs.size(-1))
            embs = embs[masks.flatten()].cpu()
            return embs, doclens

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")
        if len(passages) == 0:
            return None, None
        with torch.inference_mode():
            embs, doclens = [], []
            for passages_batch in batch(passages, self.bsize * 50):
                embs_, doclens_ = self.docFromText(
                    passages_batch, bsize=self.bsize, keep_dims='flatten', showprogress=False)
                embs.append(embs_)
                doclens.extend(doclens_)
            embs = torch.cat(embs)
        return embs, doclens
