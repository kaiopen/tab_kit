from typing import Any, Dict, List, Sequence, Self, Union
import json
from pathlib import Path

from kaitorch.pcd import PointClouds, PointCloudReaderXYZIR, PointCloudXYZIR
from kaitorch.data import mask_in_range, PI


class TAB:
    RANGE_RHO = (0, 20)
    RANGE_THETA = (-PI / 2., PI / 2.)
    RANGE_X = (0, 20)
    RANGE_Y = (-20, 20)
    RANGE_Z = (-1.5, 1.5)
    RANGE_INTENSITY = (0, 18)
    NUM_RING = 16
    GROUND = -0.73

    SEMANTICS = ('straight-going_side', 'turning')

    class Frame:
        def __init__(
            self, sequence: str, id: str, annotated: bool = False
        ) -> None:
            self._seq = sequence
            self._id: str = id
            self._anno: bool = annotated
            self._prev: Union[Self, None] = None
            self._next: Union[Self, None] = None

        @property
        def annotated(self) -> bool:
            return self._anno

        @property
        def id(self) -> str:
            return self._id

        @property
        def previous(self) -> Union[Self, None]:
            return self._prev

        @property
        def next(self) -> Union[Self, None]:
            return self._next

        @property
        def sequence(self) -> str:
            return self._seq

        def set_previous(self, f: Self) -> None:
            self._prev = f

        def set_next(self, f: Self) -> None:
            self._next = f

    def __init__(
        self, root: Union[str, Path], split: str = 'train', *args, **kwargs
    ) -> None:
        r'''

        ### Args:
            - root: path to the TAB dataset.
            - split: "train", "val", or "test".

        ### Properties:
            - ids

        ### Methods:
            - __getitem__
            - __iter__
            - __len__
            - __next__
            - get_pcd
            - get_boundaries

        ### Static Methods:
            - load_pcd
            - load_boundaries

        ### Class Methods:
            - filter_pcd_

        '''
        match split:
            case 'train' | 'val':
                f = 'trainval.json'
            case 'test':
                f = 'test.json'
            case _:
                raise ValueError(
                    'an invalid `split`. "train", "val" or "test" are expected.'
                )

        root = Path(root)

        keys = []  # annotated key frames
        for seq, fs in json.load(root.joinpath(f).open('r')).items():
            frames: List[self.Frame] = []
            for id, task in fs:
                f = self.Frame(seq, id, task is not None)
                frames.append(f)
                if task == split:
                    keys.append(f)

            if (num := len(frames)) > 1:
                frames[0].set_next(frames[1])
                frames[-1].set_previous(frames[-2])

                for i in range(1, num - 1):
                    f = frames[i]
                    f.set_previous(frames[i - 1])
                    f.set_next(frames[i + 1])
            elif 0 == num:
                continue

        self._ids = keys

        self._dir_pcd = root.joinpath('pcd')
        self._dir_bound = root.joinpath('boundary')

        self._len = len(self._ids)
        self.__i = 0

    def __getitem__(self, index: int) -> Frame:
        return self._ids[index]

    def __iter__(self):
        return self

    def __len__(self):
        return self._len

    def __next__(self) -> Frame:
        if self.__i < self._len:
            data = self[self.__i]
            self.__i += 1
            return data
        self.__i = 0
        raise StopIteration

    def get_boundaries(self, f: Frame) -> Sequence[Dict[str, Any]]:
        return self.load_boundaries(
            self._dir_bound.joinpath(f.sequence, f.id + '.json')
        )

    def get_pcd(self, f: Frame) -> PointCloudXYZIR:
        return self.load_pcd(self._dir_pcd.joinpath(f.sequence, f.id + '.pcd'))

    @staticmethod
    def load_boundaries(f: Union[Path, str]) -> Sequence[Dict[str, Any]]:
        return json.load(Path(f).open('r'))

    @staticmethod
    def load_pcd(f: Union[Path, str]) -> PointCloudXYZIR:
        return PointCloudXYZIR.from_similar(PointCloudReaderXYZIR(f))

    @classmethod
    def filter_pcd_(cls, pcd: PointClouds) -> PointClouds:
        pcd.filter_(mask_in_range(pcd.rho_, cls.RANGE_RHO))
        pcd.filter_(mask_in_range(pcd.theta_, cls.RANGE_THETA))
        pcd.filter_(mask_in_range(pcd.z_, cls.RANGE_Z))
        return pcd
