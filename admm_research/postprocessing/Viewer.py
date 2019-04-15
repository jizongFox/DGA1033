import argparse
import re
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pathlib2 import Path

Tensor = Union[np.ndarray, torch.Tensor]


def map_(func: Callable, iterator_: Iterable) -> List[Any]:
    return list(map(func, iterator_))


def get_parser() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog='3D volumne viewer for 2D-sliced images.',
        description='Group and view 2D images with different masks.',
    )
    parser.add_argument('--img_source', type=str, required=True, help='2D image source folder as the background.')
    parser.add_argument('--gt_folder', type=str, nargs='*', default=[], help='')
    return parser.parse_args()


class event_handler(object):

    def __init__(self, drawfunction: Callable) -> None:
        super().__init__()
        self.drawfunction: Callable = drawfunction

    def __call__(self, event):
        pass


class Volume(object):
    '''
    This class abstracts the necessary components, such as paired image and masks' paths, grouping paths to subjects.
    :parameter  img_folder
    :type str
    :parameter mask_folder
    :type List[str]
    :interface: __next__() and __cache__()
    :returns img matrix b h w and List[(b h w)]
    '''

    def __init__(self, img_folder: str, mask_folder_list: List[str], group_pattern=r'patient\d+_\d+',
                 img_extension: str = 'png') -> None:
        super().__init__()
        self.img_folder: Path = Path(img_folder)
        assert self.img_folder.exists(), self.img_folder
        self.mask_folder_list = [Path(m) for m in mask_folder_list]
        self.num_mask = len(self.mask_folder_list)
        for m in self.mask_folder_list:
            assert m.exists()
        self.group_pattern: str = group_pattern
        self.img_extension = img_extension
        self.img_paths, self.mask_paths_dict = self._load_images(self.img_folder, self.mask_folder_list,
                                                                 self.img_extension)
        self.img_paths_group, self.mask_paths_group_dict = self._group_images(
            self.img_paths,
            self.mask_paths_dict,
            self.group_pattern
        )
        assert self.img_paths_group.keys() == self.mask_paths_group_dict.keys()
        for subject, paths in self.img_paths_group.items():
            for m, m_paths in self.mask_paths_group_dict[subject].items():
                assert map_(lambda x: x.stem, paths) == map_(lambda x: x.stem, m_paths)
        print(f'Found {len(self.img_paths_group.keys())} subjects with totally {len(self.img_paths)} images.')
        self.identifies: List[str] = list(self.img_paths_group.keys())
        print(f'identifies: {self.identifies[:5]}...')
        self.current_identify = 0

        self.img_source, self.mask_dicts = self._preload_subjects(
            self.img_paths_group[self.identifies[self.current_identify]],
            self.mask_paths_group_dict[self.identifies[self.current_identify]])

    def __next__(self):
        if self.current_identify < len(self.identifies) - 1:
            self.current_identify += 1
            self.img_source, self.mask_dicts = self._preload_subjects(
                self.img_paths_group[self.identifies[self.current_identify]],
                self.mask_paths_group_dict[self.identifies[self.current_identify]])
        print(f'current identify num:{self.current_identify}')

        return self.img_source, self.mask_dicts, self.identifies[self.current_identify]

    def __cache__(self):
        if self.current_identify > 1:
            self.current_identify -= 1
            self.img_source, self.mask_dicts = self._preload_subjects(
                self.img_paths_group[self.identifies[self.current_identify]],
                self.mask_paths_group_dict[self.identifies[self.current_identify]])
        print(f'current identify num:{self.current_identify}')

        return self.img_source, self.mask_dicts, self.identifies[self.current_identify]

    @staticmethod
    def _load_images(img_folder: Path, mask_folder_list: List[Path], img_extension: str) -> \
            Tuple[List[Path], Dict[str, List[Path]]]:
        img_paths: List[Path] = list(img_folder.rglob(f'**/*.{img_extension.replace(".", "")}'))
        assert len(img_paths) > 0, f'The length of the image must be higher than 1, given {len(img_paths)}.'
        mask_paths_dict: Dict[str, List[Path]] = {}
        for m in mask_folder_list:
            mask_paths_dict[str(m)] = list(m.rglob(f'**/*.{img_extension.replace(".", "")}'))
            assert mask_paths_dict[str(m)].__len__() > 0, f'The length of the masks {m} must be higher than 1, \
            given {mask_paths_dict[str(m)].__len__()}'
        assert len(img_paths) == list(mask_paths_dict.values())[0].__len__()
        assert len(set(map(len, list(mask_paths_dict.values())))) == 1, len(
            set(map(len, list(mask_paths_dict.values()))))
        return img_paths, mask_paths_dict

    @staticmethod
    def _group_images(img_paths: List[Path],
                      mask_paths_dict: Dict[str, List[Path]],
                      group_pattern: str) \
            -> Tuple[Dict[str, List[Path]], Dict[str, Dict[str, List[Path]]]]:
        img_stem: List[str] = [i.stem for i in img_paths]
        r_pattern = re.compile(group_pattern)
        img_identification = sorted([r_pattern.match(i_stem).group(0) for i_stem in img_stem])  # type: ignore
        unique_identification = sorted(list(set(img_identification)))
        # grouping
        img_paths_group: Dict[str, List[Path]] = {}
        mask_paths_group: Dict[str, Dict[str, List[Path]]] = {}
        for identy in unique_identification:
            img_paths_group[identy] = sorted([i for i in img_paths if re.compile(identy).search(str(i)) is not None])
            mask_paths_group[identy] = {}
            for m, m_list in mask_paths_dict.items():
                mask_paths_group[identy][m] = sorted(
                    [i for i in m_list if re.compile(identy).search(str(i)) is not None])
        return img_paths_group, mask_paths_group

    @staticmethod
    def _preload_subjects(batch_image_path: List[Path], batch_mask_paths_dict: Dict[str, List[Path]]) \
            -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        img_source: np.ndarray = np.stack([np.array(Image.open(str(x)).convert('P')) for x in batch_image_path], axis=0)
        mask_source_dict: Dict[str, np.ndarray] = {}
        for k, v in batch_mask_paths_dict.items():
            mask_source_dict[k] = np.stack([np.array(Image.open(str(x)).convert('P')) for x in v], axis=0)
            assert mask_source_dict[k].shape == img_source.shape
        return img_source, mask_source_dict


class Multi_Slice_Viewer(object):

    def __init__(self, volume: Volume, n_subject=1, shuffle_subject=False) -> None:
        super().__init__()
        self.volume = volume
        self.n_subject = n_subject
        self.volume.identifies = np.random.permutation(
            self.volume.identifies) if shuffle_subject else self.volume.identifies

    def _preproccess_data(self, volume_output):
        img_volume, gt_volume_dict, subject_name = volume_output
        mask_volumes = list(gt_volume_dict.values())
        _, _, _ = img_volume.shape
        if not isinstance(mask_volumes, list):
            mask_volumes = [mask_volumes]
        if mask_volumes[0] is not None:
            assert img_volume.shape == mask_volumes[0].shape
        return img_volume, mask_volumes, subject_name

    def show(self, ):
        fig, axs = plt.subplots(self.n_subject, self.volume.num_mask)
        axs = np.array([axs]) if not isinstance(axs, np.ndarray) else axs
        self.axs = axs if len(axs.shape) == 2 else axs[None, ...]

        for row in range(self.n_subject):
            img_volume, mask_volumes, subject_name = self._preproccess_data(self.volume.__next__())
            for i, (ax, mask_volume) in enumerate(zip(self.axs[row], mask_volumes)):
                ax.subject_name = subject_name
                ax.mask_volume = mask_volume
                ax.img_volume = img_volume
                ax.index = img_volume.shape[0] // 2
                ax.imshow(ax.img_volume[ax.index], cmap='gray')
                if mask_volume is not None:
                    ax.con = ax.contour(ax.mask_volume[ax.index])
                ax.axis('off')
                ax.set_title(f'{subject_name} @ plane:{ax.index}')

        fig.canvas.mpl_connect('scroll_event', self.process_mouse_wheel)
        fig.canvas.mpl_connect('button_press_event', self.process_mouse_button)
        # plt.tight_layout()
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show()

    def process_mouse_wheel(self, event):
        fig = event.canvas.figure
        for i, ax in enumerate(fig.axes):
            if event.button == 'up':
                self._previous_slice(ax)
            elif event.button == 'down':
                self._next_slice(ax)
        fig.canvas.draw()

    def process_mouse_button(self, event):
        fig = event.canvas.figure
        if event.button in (1, 3):
            if event.button == 1:
                self._change_subject(mode=self.volume.__next__)
            elif event.button == 3:
                self._change_subject(mode=self.volume.__cache__)
            fig.canvas.draw()

    def _change_subject(self, mode):
        axs = self.axs
        for row in range(self.n_subject):
            img_volume, mask_volumes, subject_name = self._preproccess_data(mode())
            for i, (ax, mask_volume) in enumerate(zip(axs[row], mask_volumes)):
                ax.clear()
                ax.subject_name = subject_name
                ax.mask_volume = mask_volume
                ax.img_volume = img_volume
                ax.index = img_volume.shape[0] // 2
                ax.imshow(ax.img_volume[ax.index], cmap='gray')
                if mask_volume is not None:
                    ax.con = ax.contour(ax.mask_volume[ax.index])
                ax.axis('off')
                ax.set_title(f'{subject_name} @ plane:{ax.index}')

    @staticmethod
    def _previous_slice(ax):
        img_volume = ax.img_volume
        if ax.mask_volume is not None:
            for con in ax.con.collections:
                try:
                    con.remove()
                except:
                    pass
        ax.index = (ax.index - 1) if (ax.index - 1) >= 0 else 0  # wrap around using %
        ax.images[0].set_array(img_volume[ax.index])
        if ax.mask_volume is not None:
            ax.con = ax.contour(ax.mask_volume[ax.index])
        ax.set_title(f'{ax.subject_name} @ plane:{ax.index}')

    @staticmethod
    def _next_slice(ax):
        img_volume = ax.img_volume
        if ax.mask_volume is not None:
            for con in ax.con.collections:
                try:
                    con.remove()
                except:
                    pass
        ax.index = (ax.index + 1) if (ax.index + 1) < img_volume.shape[0] else img_volume.shape[0] - 1
        ax.images[0].set_array(img_volume[ax.index])
        if ax.mask_volume is not None:
            ax.con = ax.contour(ax.mask_volume[ax.index])
        ax.set_title(f'plane = {ax.index}')
        ax.set_title(f'{ax.subject_name} @ plane:{ax.index}')


if __name__ == '__main__':
    V = Volume(
        '../../admm_research/dataset/ACDC-2D-All/val/Img',
        ['../../admm_research/dataset/ACDC-2D-All/val/GT',
         '../../archives/ACDC/size/iter1000/best',
         '../../archives/ACDC/gc_size/iter1000/best',
         '../../archives/ACDC/fs/iter000/best']
    )

    Viewer = Multi_Slice_Viewer(V, shuffle_subject=True, n_subject=3)
    Viewer.show()
