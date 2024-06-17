from typing import List, Tuple

import torch

from kaitorch.typing import TorchTensor, TorchFloat


class KMMatch:
    def __init__(self) -> None:
        self._weights = None
        self._num_l = self._num_r = 0
        self._w_l = self._w_r = None
        self._m_l = self._m_r = None
        self._o_l = self._o_r = None
        self._res_0 = self._res_1 = None
        self._qwe = 0

    def __call__(
        self, weights: TorchTensor[TorchFloat]
    ) -> Tuple[List[int], List[int]]:
        self._num_l, self._num_r = weights.shape
        if self._num_l > self._num_r:
            weights = weights.T
            self._num_l, self._num_r = weights.shape
            self._res_1 = self._m_l = [-1 for _ in range(self._num_l)]
            self._res_0 = self._m_r = [-1 for _ in range(self._num_r)]
        else:
            self._res_0 = self._m_l = [-1 for _ in range(self._num_l)]
            self._res_1 = self._m_r = [-1 for _ in range(self._num_r)]

        # print('weights:')
        # print(weights)
        self._w_l = torch.max(weights, dim=1)[0].tolist()
        self._w_r = [0 for _ in range(self._num_r)]
        self._weights = weights.tolist()
        # print('w_l:')
        # print(self._w_l)
        # print('w_r:')
        # print(self._w_r)

        for u in range(self._num_l):
            # print(f'for {u}')
            if self._w_l[u] <= 0:
                # print('w_l[u] <= 0 -> continue')
                continue

            while True:
                # print('while')
                self._o_l = [False for _ in range(self._num_l)]
                self._o_r = [False for _ in range(self._num_r)]

                if self._match(u):
                    # print('matched')
                    # print('m_l')
                    # print(self._m_l)
                    # print('m_r')
                    # print(self._m_r)
                    break

                # print('not matched')
                d = 1000
                for i in range(self._num_l):
                    if (self._o_l[i]):
                        for j in range(self._num_r):
                            if (
                                not self._o_r[j]
                                and (w := self._weights[i][j]) > 0
                            ):
                                d = min(d, self._w_l[i] + self._w_r[j] - w)
                if d > 999:
                    return self._res_0, self._res_1

                for i in range(self._num_l):
                    if self._o_l[i]:
                        self._w_l[i] -= d
                for j in range(self._num_r):
                    if self._o_r[j]:
                        self._w_r[j] += d
                # print('w_l')
                # print(self._w_l)
                # print('w_r')
                # print(self._w_r)
        return self._res_0, self._res_1

    def _match(self, i: int) -> bool:
        # print(f'match {i}')
        self._o_l[i] = True
        w_l = self._w_l[i]
        for j in range(self._num_r):
            if (
                not self._o_r[j]
                and (w := self._weights[i][j]) > 0
                and abs(w_l + self._w_r[j] - w) < 1e-6
            ):
                self._o_r[j] = True
                if -1 == (k := self._m_r[j]) or self._match(k):
                    self._m_l[i] = j
                    self._m_r[j] = i
                    return True
        return False
