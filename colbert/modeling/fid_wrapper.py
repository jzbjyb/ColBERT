import torch
from transformers import T5Tokenizer
import sys
from pathlib import Path
sys.path.insert(0, f'{Path.home()}/exp/FiD')
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
                 half_precision: bool = True,
                 normalize: bool = True):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', return_dict=False)
        self.model = FiDT5.from_pretrained(model_path)
        self.model.eval()
        self.doc_maxlen = doc_maxlen
        self.query_maxlen = query_maxlen
        self.head_idx = head_idx
        self.bsize = bsize
        self.half_precision = half_precision
        self.normalize = normalize
        self.query_tokenizer = type('placeholder', (object,), {'query_maxlen': None})()

    def cuda(self):
        self.model.cuda()
        return self
    
    def get_tokenizer(self, type: str = 'query'):
        if type == 'query':
            return self.tokenizer
        elif type == 'doc':
            return self.tokenizer
        raise ValueError

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None, return_more: bool = False):
        if context is not None:
            raise NotImplementedError
        if bsize is None:
            bsize = len(queries)
        with torch.no_grad():
            embs, idss, masks = [], [], []
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
                idss.append(ids)
                masks.append(mask)
            embs = torch.cat(embs)
            idss = torch.cat(idss)
            masks = torch.cat(masks)
            if to_cpu:
                embs = embs.cpu()
                idss = idss.cpu()
                masks = masks.cpu()
            if return_more:
                return embs, idss, masks
            return embs

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        if keep_dims != 'flatten' or to_cpu != False or showprogress != False:
            raise NotImplementedError
        if bsize is None:
            bsize = len(docs)
        with torch.no_grad():
            embs, masks, tokens = [], [], []
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
                tokens.append(ids)
            embs, masks, tokens = torch.cat(embs), torch.cat(masks), torch.cat(tokens)
            doclens = masks.sum(-1).tolist()
            embs = embs.view(-1, embs.size(-1))
            embs = embs[masks.flatten()].cpu()
            tokens = tokens.flatten()[masks.flatten()].cpu()
            if return_tokens:
                return embs, doclens, tokens
            else:
                return embs, doclens

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")
        if len(passages) == 0:
            return None, None
        with torch.inference_mode():
            embs, doclens, tokens = [], [], []
            for passages_batch in batch(passages, self.bsize * 50):
                embs_, doclens_, tokens_ = self.docFromText(
                    passages_batch, bsize=self.bsize, keep_dims='flatten', showprogress=False, return_tokens=True)
                embs.append(embs_)
                doclens.extend(doclens_)
                tokens.append(tokens_)
            embs = torch.cat(embs)
            tokens = torch.cat(tokens)
        return embs, doclens, tokens
