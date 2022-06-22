from struct import pack
import torch
from torch._C import device

from colbert.utils.utils import flatten

from .strided_tensor_core import StridedTensorCore, _create_mask, _create_view


"""
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
"""


class StridedTensor(StridedTensorCore):
    def __init__(self, packed_tensor, lengths, dim=None):
        super().__init__(packed_tensor, lengths, dim=dim)

    @classmethod
    def pad_packed(cls, packed_tensor, lengths):
        assert False, "This seems to be incorrect but I can't see why. Is it the inner_dims in the views?"

        packed_tensor, lengths = packed_tensor.cuda().contiguous(), lengths.cuda()

        inner_dims = packed_tensor.size()[1:]
        stride = lengths.max().item()
        offsets = torch.cumsum(lengths, dim=0) - lengths[0]

        padding = torch.zeros(stride, *inner_dims, device=packed_tensor.device, dtype=packed_tensor.dtype)
        packed_tensor = torch.cat((packed_tensor, padding))

        view = _create_view(packed_tensor, stride, inner_dims)[offsets]
        mask = _create_mask(lengths, stride, like=view)

        return view, mask

    def _prepare_lookup(self, pids):
        if isinstance(pids, list):
            pids = torch.tensor(pids)

        assert pids.dim() == 1

        pids = pids.cuda().long()
        lengths = self.lengths[pids].cuda()
        offsets = self.offsets[pids]

        return pids, lengths, offsets

    def lookup(self, pids, output='packed', skip_largest_view: bool = False):
        largest_view = self.strides[-1]
        second_largest_view = self.strides[-2]
        pids, lengths, offsets = self._prepare_lookup(pids)
        if skip_largest_view:
            skip_mask = lengths <= second_largest_view
            pids = pids[skip_mask]
            lengths = lengths[skip_mask]
            offsets = offsets[skip_mask]

        stride = lengths.max().item()
        stride = next(s for s in self.strides if stride <= s)

        tensor = self.views[stride][offsets].cuda()

        mask = _create_mask(lengths, stride)

        if output == 'padded':
            return tensor, mask

        assert output == 'packed'

        tensor = tensor[mask]

        return tensor, lengths

    def lookup_staggered(self, pids, output='packed'):
        permute_idxs, unordered_tensors, unordered_lengths, unordered_masks = self.lookup_packed_unordered(pids)

        output_tensor = torch.empty(permute_idxs.size(0), self.max_stride, *self.inner_dims,
                                    dtype=unordered_tensors[0].dtype, device=unordered_tensors[0].device)

        output_mask = torch.zeros(permute_idxs.size(0), self.max_stride,
                                  dtype=unordered_masks[0].dtype, device=unordered_masks[0].device)

        offset = 0
        for tensor, mask in zip(unordered_tensors, unordered_masks):
            endpos = offset + tensor.size(0)
            output_tensor[offset:endpos, :tensor.size(1)] = tensor
            output_mask[offset:endpos, :mask.size(1)] = mask
            offset = endpos

        output_mask = output_mask[permute_idxs]
        output_tensor = output_tensor[permute_idxs]

        if output == 'padded':
            return output_tensor, output_mask

        assert output == 'packed'

        output_tensor = output_tensor[output_mask]

        return output_tensor, unordered_lengths[permute_idxs]

    def lookup_packed_unordered(self, pids):
        pids, lengths, offsets = self._prepare_lookup(pids)

        lengths2 = lengths.clone()
        sentinel = self.strides[-1] + 1
        order = torch.arange(pids.size(0), device='cuda')

        all_orders = []
        all_tensors = []
        all_lengths = []
        all_masks = []

        for stride in self.strides:
            is_shorter = lengths2 <= stride

            if is_shorter.sum() == 0:
                continue

            order_ = order[is_shorter]
            tensor_, lengths_, mask_ = self._lookup_with_stride(stride, lengths[is_shorter], offsets[is_shorter])

            all_orders.append(order_)
            all_tensors.append(tensor_)
            all_lengths.append(lengths_)
            all_masks.append(mask_)

            lengths2[is_shorter] = sentinel

        assert lengths2.allclose(torch.tensor([sentinel], device='cuda'))

        all_orders = torch.cat(all_orders)
        permute_idxs = torch.sort(all_orders).indices

        return permute_idxs, all_tensors, torch.cat(all_lengths), all_masks

    def _lookup_with_stride(self, stride, lengths, offsets):
        tensor = self.views[stride][offsets].cuda()

        mask = _create_mask(lengths, stride)
        # tensor = tensor[mask]

        return tensor, lengths, mask


if __name__ == '__main__':
    # lst = []
    # for _ in range(10):
    #     lst.append(list(range(random.randint(0, 10))))

    # print(lst)

    # t = StridedTensor.from_nested_list(lst)
    # print(t.lookup([9]))

    import os
    import pickle

    index_path = '/future/u/okhattab/root/unit/indexes/2021/08/residual.NQ-micro'
    with open(os.path.join(index_path, "centroid_idx_to_embedding_ids.pickle"), "rb") as f:
        ivf_list = pickle.load(f)

    assert len(ivf_list) == max(ivf_list.keys()) + 1
    ivf_list = [ivf_list[i] for i in range(len(ivf_list))]

    for x in ivf_list:
        assert type(x) is list
        assert type(x[0]) is int

    ncentroids = len(ivf_list)

    ivf = StridedTensor.from_nested_list(ivf_list)

    import time

    torch.cuda.synchronize()
    t = time.time()

    N = 100
    for _ in range(N):
        probed_centroids = torch.randint(0, ncentroids, size=(32, 8)).flatten()
        emb_ids, emb_ids_lengths = ivf.lookup(probed_centroids).as_packed_tensor()

    torch.cuda.synchronize()
    print((time.time() - t) * 1000 / N, 'ms')

    print(emb_ids_lengths)

    slow_result = flatten([ivf_list[idx] for idx in probed_centroids.flatten().tolist()])
    print(emb_ids.size(), len(slow_result))

    for a, b in zip(slow_result, emb_ids.flatten().tolist()):
        assert a == b, (a, b)

    print("#> Done!")

    print(ivf.lookup(probed_centroids).as_padded_tensor()[0].size())
