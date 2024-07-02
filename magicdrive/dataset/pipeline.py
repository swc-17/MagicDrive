import os
import logging
import warnings
from typing import Any, Dict, Tuple

import h5py
import numpy as np
from numpy import random
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from PIL import Image
import PIL.ImageDraw as ImageDraw

import torch
import torchvision

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet.datasets.pipelines import LoadAnnotations

# from mmdet3d.core.bbox import (
#     CameraInstance3DBoxes,
#     DepthInstance3DBoxes,
#     LiDARInstance3DBoxes,
# )
from magicdrive.core.bbox_structure.base_box3d import BaseInstance3DBoxes
from magicdrive.core.bbox_structure.lidar_box3d import LiDARInstance3DBoxes
from .pipeline_utils import one_hot_decode


@PIPELINES.register_module()
class LoadBEVSegmentationM:
    '''This only loads map annotations, there is no dynamic objects.
    In this map, the origin is at lower-left corner, with x-y transposed.
                          FRONT                             RIGHT
         Nuscenes                       transposed
        --------->  LEFT   EGO   RIGHT  ----------->  BACK   EGO   FRONT
           map                            output
                    (0,0)  BACK                       (0,0)  LEFT
    Guess reason, in cv2 / PIL coord, this is a BEV as follow:
        (0,0)  LEFT

        BACK   EGO   FRONT

              RIGHT
    All masks in np channel first format.
    '''

    AUX_DATA_CH = {
        "visibility": 1,
        "center_offset": 2,
        "center_ohw": 4,
        "height": 1,
    }

    def __init__(
        self,
        dataset_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        classes: Tuple[str, ...],
        object_classes: Tuple[str, ...] = None,  # object_classes
        aux_data: Tuple[str, ...] = None,  # aux_data for dynamic objects
        cache_file: str = None,
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        canvas_h = int(patch_h / ybound[2])
        canvas_w = int(patch_w / xbound[2])
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.classes = classes
        self.object_classes = object_classes
        self.aux_data = aux_data
        self.lidar2canvas = np.array([
            [canvas_h / patch_h, 0, canvas_h / 2],
            [0, canvas_w / patch_w, canvas_w / 2],
            [0, 0, 1]
        ])

        self.maps = {}
        for location in LOCATIONS:
            self.maps[location] = NuScenesMap(dataset_root, location)
        if cache_file and os.path.isfile(cache_file):
            logging.info(f"using data cache from: {cache_file}")
            # load to memory and ignore all possible changes.
            self.cache = cache_file
        else:
            self.cache = None
        # this should be set through main process afterwards
        self.shared_mem_cache = None

    def _get_dynamic_aux_bbox(self, aux_mask, data: Dict[str, Any]):
        '''Three aux data (7 channels in total), class-agnostic:
        1. visibility, 1-channel
        2. center-offset, 2-channel
        3. height/2, width/2, orientation, 4-channel, on bev canvas
        4. height of bbox, in lidar coordinate
        '''
        for _idx in range(len(data['gt_bboxes_3d'])):
            box = data['gt_bboxes_3d'][_idx]
            # get canvas coordinates
            # fmt:off
            _box_lidar = np.concatenate([
                box.corners[:, [0, 3, 7, 4], :2].numpy(),
                box.bottom_center[:, None, :2].numpy(),  # center
                box.corners[:, [4, 7], :2].mean(dim=1)[:, None].numpy(),  # front
                box.corners[:, [0, 4], :2].mean(dim=1)[:, None].numpy(),  # left
            ], axis=1)
            # fmt:on
            _box_canvas = np.dot(
                np.pad(_box_lidar, ((0, 0), (0, 0), (0, 1)), constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # in canvas coordinates
            box_canvas = _box_canvas[0, :4]
            center_canvas = _box_canvas[0, 4:5]
            front_canvas = _box_canvas[0, 5:6]
            left_canvas = _box_canvas[0, 6:7]
            # render mask
            render = Image.fromarray(np.zeros(self.canvas_size, dtype=np.uint8))
            draw = ImageDraw.Draw(render)
            draw.polygon(
                box_canvas.round().astype(np.int32).flatten().tolist(),
                fill=1)
            # construct
            tmp_mask = np.array(render) > 0
            coords = np.stack(np.meshgrid(
                np.arange(self.canvas_size[1]), np.arange(self.canvas_size[0])
            ), -1).astype(np.float32)
            _cur_ch = 0
            if "visibility" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['visibility']
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = data['visibility'][_idx]
                _cur_ch = _ch_stop
            if "center_offset" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_offset']
                center_offset = coords[tmp_mask] - center_canvas
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = center_offset
                _cur_ch = _ch_stop
            if "center_ohw" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['center_ohw']
                height = np.linalg.norm(front_canvas - center_canvas)
                width = np.linalg.norm(left_canvas - center_canvas)
                # yaw = box.yaw  # scaling aspect ratio, yaw does not change
                v = ((front_canvas - center_canvas) / (
                    np.linalg.norm(front_canvas - center_canvas) + 1e-6))[0]
                # yaw = - np.arctan2(v[1], v[0])  # add negative, align with mmdet coord
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    height, width, v[0], v[1]])[None]
                _cur_ch = _ch_stop
            if "height" in self.aux_data:
                _ch_stop = _cur_ch + self.AUX_DATA_CH['height']
                bbox_height = box.height.item()  # in lidar coordinate
                aux_mask[tmp_mask, _cur_ch:_ch_stop] = np.array([
                    bbox_height])[None]
                _cur_ch = _ch_stop
        return aux_mask

    def _get_dynamic_aux(self, data: Dict[str, Any] = None) -> Any:
        '''aux data
        case 1: self.aux_data is None, return None
        case 2: data=None, set all values to zeros
        '''
        if self.aux_data is None:
            return None  # there is no aux_data

        aux_ch = sum([self.AUX_DATA_CH[aux_k] for aux_k in self.aux_data])
        if aux_ch == 0:  # there is no available aux_data
            if len(self.aux_data) != 0:
                logging.warn(f"Your aux_data: {self.aux_data} is not available")
            return None

        aux_mask = np.zeros((*self.canvas_size, aux_ch), dtype=np.float32)
        if data is not None:
            aux_mask = self._get_dynamic_aux_bbox(aux_mask, data)

        # transpose x,y and channel first format
        aux_mask = aux_mask.transpose(2, 1, 0)
        return aux_mask

    def _project_dynamic_bbox(self, dynamic_mask, data):
        '''We use PIL for projection, while CVT use cv2. The results are
        slightly different due to anti-alias of line, but should be similar.
        '''
        for cls_id, cls_name in enumerate(self.object_classes):
            # pick boxes
            cls_mask = data['gt_labels_3d'] == cls_id
            boxes = data['gt_bboxes_3d'][cls_mask]
            if len(boxes) < 1:
                continue
            # get coordinates on canvas. the order of points matters.
            bottom_corners_lidar = boxes.corners[:, [0, 3, 7, 4], :2]
            bottom_corners_canvas = np.dot(
                np.pad(bottom_corners_lidar.numpy(), ((0, 0), (0, 0), (0, 1)),
                       constant_values=1.0),
                self.lidar2canvas.T)[..., :2]  # N, 4, xy
            # draw
            render = Image.fromarray(dynamic_mask[cls_id])
            draw = ImageDraw.Draw(render)
            for box in bottom_corners_canvas:
                draw.polygon(
                    box.round().astype(np.int32).flatten().tolist(), fill=1)
            # save
            dynamic_mask[cls_id, :] = np.array(render)[:]
        return dynamic_mask

    def _project_dynamic(self, static_label, data: Dict[str, Any]) -> Any:
        """for dynamic mask, one class per channel
        case 1: data is None, set all values to zeros
        """
        # setup
        ch = len(self.object_classes)
        dynamic_mask = np.zeros((ch, *self.canvas_size), dtype=np.uint8)

        # if int, set ch=object_classes with all zeros; otherwise, project
        if data is not None:
            dynamic_mask = self._project_dynamic_bbox(dynamic_mask, data)

        # combine with static_label
        dynamic_mask = dynamic_mask.transpose(0, 2, 1)
        combined_label = np.concatenate([static_label, dynamic_mask], axis=0)
        return combined_label

    def _load_from_cache(
            self, data: Dict[str, Any], cache_dict) -> Dict[str, Any]:
        token = data['token']
        labels = one_hot_decode(
            cache_dict['gt_masks_bev_static'][token][:], len(self.classes))
        if self.object_classes is not None:
            if None in self.object_classes:
                # HACK: if None, set all values to zero
                # there is no computation, we generate on-the-fly
                final_labels = self._project_dynamic(labels, None)
                aux_labels = self._get_dynamic_aux(None)
            else:  # object_classes is list, we can get from cache_file
                final_labels = one_hot_decode(
                    cache_dict['gt_masks_bev'][token][:],
                    len(self.classes) + len(self.object_classes)
                )
                aux_labels = cache_dict['gt_aux_bev'][token][:]
            data["gt_masks_bev_static"] = labels
            data["gt_masks_bev"] = final_labels
            data["gt_aux_bev"] = aux_labels
        else:
            data["gt_masks_bev_static"] = labels
            data["gt_masks_bev"] = labels
        return data

    def _get_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        lidar2ego = data["lidar2ego"]
        ego2global = data["ego2global"]
        # lidar2global = ego2global @ lidar2ego @ point2lidar
        lidar2global = ego2global @ lidar2ego
        if "lidar_aug_matrix" in data:  # it is I if no lidar aux or no train
            lidar2point = data["lidar_aug_matrix"]
            point2lidar = np.linalg.inv(lidar2point)
            lidar2global = lidar2global @ point2lidar

        map_pose = lidar2global[:2, 3]
        patch_box = (
            map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])

        rotation = lidar2global[:3, :3]
        v = np.dot(rotation, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])  # angle between v and x-axis
        patch_angle = yaw / np.pi * 180

        mappings = {}
        for name in self.classes:
            if name == "drivable_area*":
                mappings[name] = ["road_segment", "lane"]
            elif name == "divider":
                mappings[name] = ["road_divider", "lane_divider"]
            else:
                mappings[name] = [name]

        layer_names = []
        for name in mappings:
            layer_names.extend(mappings[name])
        layer_names = list(set(layer_names))

        # cut semantics from nuscenesMap
        location = data["location"]
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=patch_angle,
            layer_names=layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)  # TODO why need transpose here?
        masks = masks.astype(np.bool)

        # here we handle possible combinations of semantics
        num_classes = len(self.classes)
        labels = np.zeros((num_classes, *self.canvas_size), dtype=np.long)
        for k, name in enumerate(self.classes):
            for layer_name in mappings[name]:
                index = layer_names.index(layer_name)
                labels[k, masks[index]] = 1

        if self.object_classes is not None:
            data["gt_masks_bev_static"] = labels
            final_labels = self._project_dynamic(labels, data)
            aux_labels = self._get_dynamic_aux(data)
            data["gt_masks_bev"] = final_labels
            data["gt_aux_bev"] = aux_labels
        else:
            data["gt_masks_bev_static"] = labels
            data["gt_masks_bev"] = labels
        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # if set cache, use it.
        if self.shared_mem_cache:
            try:
                return self._load_from_cache(data, self.shared_mem_cache)
            except:
                pass
        if self.cache:
            try:
                with h5py.File(self.cache, 'r') as cache_file:
                    return self._load_from_cache(data, cache_file)
            except:
                pass

        # cache miss, load normally
        data = self._get_data(data)

        # if set, add this item into it.
        if self.shared_mem_cache:
            token = data['token']
            for key in self.shared_mem_cache.keys():
                self.shared_mem_cache[key][token] = data[key]
        return data


@PIPELINES.register_module()
class ObjectRangeFilterM:
    """Filter objects by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter objects by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(
            data["gt_bboxes_3d"], (LiDARInstance3DBoxes,)
        ):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        # elif isinstance(data["gt_bboxes_3d"], CameraInstance3DBoxes):
        #     bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = data["gt_bboxes_3d"]
        gt_labels_3d = data["gt_labels_3d"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        data["gt_bboxes_3d"] = gt_bboxes_3d
        data["gt_labels_3d"] = gt_labels_3d
        if "visibility" in data:
            data["visibility"] = data["visibility"][
                mask.numpy().astype(np.bool)]

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str


@PIPELINES.register_module()
class ReorderMultiViewImagesM:
    """Reorder camera views.
    ori_order is from `tools/data_converter/nuscenes_converter.py`
    Args:
        order (list[str]): List of camera names.
        safe (bool, optional): if True, will check every key. Defaults to True.
    """

    ORI_ORDER = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    SAFE_KEYS = [
        "token",
        "sample_idx",
        "lidar_path",
        "sweeps",
        "timestamp",
        "location",
        "description",
        "timeofday",
        "visibility",
        "ego2global",
        "lidar2ego",
        "ann_info",
        "img_fields",
        "bbox3d_fields",
        "pts_mask_fields",
        "pts_seg_fields",
        "bbox_fields",
        "mask_fields",
        "seg_fields",
        "box_type_3d",
        "box_mode_3d",
        "img_shape",
        "ori_shape",
        "pad_shape",
        "scale_factor",
        "gt_bboxes_3d",
        "gt_labels_3d",
        "lidar_aug_matrix",
        "gt_masks_bev_static",
        "gt_masks_bev",
        "gt_aux_bev",
    ]
    REORDER_KEYS = [
        "image_paths",
        "lidar2camera",
        "lidar2image",
        "camera2ego",
        "camera_intrinsics",
        "camera2lidar",
        "filename",
        "img",
        "img_aug_matrix",
    ]
    WARN_KEYS = []

    def __init__(self, order, safe=True):
        self.order = order
        self.order_mapper = [self.ORI_ORDER.index(it) for it in self.order]
        self.safe = safe

    def reorder(self, value):
        assert len(value) == len(self.order_mapper)
        if isinstance(value, list):  # list do not support indexing by list
            return [value[i] for i in self.order_mapper]
        else:
            return value[self.order_mapper]

    def __call__(self, data):
        if self.safe:
            for k in [k for k in data.keys()]:
                if k in self.SAFE_KEYS:
                    continue
                if k in self.REORDER_KEYS:
                    data[k] = self.reorder(data[k])
                elif k in self.WARN_KEYS:
                    # it should be empty or none
                    assert not data[k], f"you need to handle {k}: {data[k]}"
                else:
                    raise RuntimeWarning(
                        f"You have unhandled key ({k}) in data")
        else:
            for k in self.REORDER_KEYS:
                if k in data:
                    data[k] = self.reorder(data[k])
        return data


@PIPELINES.register_module()
class ObjectNameFilterM:
    """Filter GT objects by their names.
    As object names in use are assigned by initialization, this only remove -1,
    i.e., unknown / unmapped classes.
    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, data):
        gt_labels_3d = data["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        data["gt_bboxes_3d"] = data["gt_bboxes_3d"][gt_bboxes_mask]
        data["gt_labels_3d"] = data["gt_labels_3d"][gt_bboxes_mask]
        if "visibility" in data:
            data["visibility"] = data["visibility"][gt_bboxes_mask]
        return data


@PIPELINES.register_module()
class RandomFlip3DwithViews:
    """consider ori_order from
    `bevfusion/tools/data_converter/nuscenes_converter.py`, as follows:
        ORI_ORDER = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
    We also assume images views have same size (ori & resized).
    """
    SUPPORT_TYPE = [None, 'v', 'h', 'handv', 'horv', 'hv']
    REORDER_KEYS = [
        "image_paths",
        "filename",
        "img",
        "camera_intrinsics",
        "camera2lidar",
        # "lidar2camera",
        # "lidar2image",
        # "camera2ego",
    ]
    IMG_ORI_SIZE = [1600, 900]

    VERTICLE_FLIP_ORDER = [0, 2, 1, 3, 5, 4]  # see the order above
    HORIZEONTAL_FLIP_ORDER = [3, 5, 4, 0, 2, 1]  # see the order above

    def __init__(self, flip_ratio, direction='v', reorder=True) -> None:
        """random flip bbox, bev, points, image views

        Args:
            flip_ratio (float): prob to flip. 1 means always, 0 means never.
            direction (str, optional): h (front-back) or v (left-right).
            Defaults to 'v'.
            reorder (bool, optional): whether reorder & flip camera view.
        """
        assert 0 <= flip_ratio <= 1, f"flip ratio in [0,1]. You provide {flip_ratio}"
        assert direction in self.SUPPORT_TYPE, f"direction should from {self.SUPPORT_TYPE}"
        if not reorder:
            warnings.warn(f"You should always use reorder, please check!")
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.reorder = reorder
        logging.info(
            f"[RandomFlip3DwithViews] ratio={self.flip_ratio}, "
            f"direction={self.direction}, reorder={self.reorder}")

    def _reorder_func(self, value, order):
        assert len(value) == len(order)
        if isinstance(value, list):  # list do not support indexing by list
            return [value[i] for i in order]
        else:
            return value[order]

    def reorder_data(self, data, order):
        # flip camera views
        if "img" in data:
            new_imgs = []
            for img in data['img']:
                new_imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            data['img'] = new_imgs
        # change ordering, left <-> right / left-front <-> right-back
        for k in self.REORDER_KEYS:
            if k in data:
                data[k] = self._reorder_func(
                    data[k], order)
        # if flip, x offset should be reversed according to image width
        if "camera_intrinsics" in data:
            params = []
            for cam_i in data['camera_intrinsics']:
                cam_i = cam_i.copy()
                cam_i[0, 2] = self.IMG_ORI_SIZE[0] - cam_i[0, 2]
                params.append(cam_i)
            data['camera_intrinsics'] = params
        return data

    def flip_vertical(self, data, rotation):
        # rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
        if "points" in data:
            data["points"].flip("vertical")
        if "gt_bboxes_3d" in data:
            data["gt_bboxes_3d"].flip("vertical")
        if "gt_masks_bev" in data:
            data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()
        # change camera extrinsics, camera2lidar is the axis rotation from lidar
        # to camera, we use moving axis transformations.
        if "camera2lidar" in data:
            params = []
            for c2l in data['camera2lidar']:
                c2l = np.array([  # flip x about lidar
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]) @ c2l.copy()
                # if not reorder, flipping axis ends up with left-handed
                # coordinate.
                if self.reorder:
                    c2l = c2l @ np.array([  # flip y about new axis
                        [1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ]) @ np.array([  # rotz 180 degree about new axis
                        [-1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ])
                params.append(c2l)
            data['camera2lidar'] = params
        if self.reorder:
            data = self.reorder_data(data, self.VERTICLE_FLIP_ORDER)
        return data, rotation

    def flip_horizontal(self, data, rotation):
        # rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
        if "points" in data:
            data["points"].flip("horizontal")
        if "gt_bboxes_3d" in data:
            data["gt_bboxes_3d"].flip("horizontal")
        if "gt_masks_bev" in data:
            data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()
        # change camera extrinsics, camera2lidar is the axis rotation from lidar
        # to camera, we use moving axis transformations.
        if "camera2lidar" in data:
            params = []
            for c2l in data['camera2lidar']:
                c2l = np.array([  # flip y about lidar
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]) @ c2l.copy()
                # if not reorder, flipping axis ends up with left-handed
                # coordinate.
                if self.reorder:
                    c2l = c2l @ np.array([  # flip x about new axis
                        [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ])
                params.append(c2l)
            data['camera2lidar'] = params
        if self.reorder:
            data = self.reorder_data(data, self.HORIZEONTAL_FLIP_ORDER)
        return data, rotation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        flip = random.rand() < self.flip_ratio
        if not flip or self.direction is None:
            return data

        rotation = np.eye(3)
        if self.direction == "horv":
            directions = random.choice(['h', 'v'], 1)
        elif self.direction == "handv":
            directions = ['h', 'v']
        elif self.direction == "hv":
            choices = [['h'], ['v'], ['h', 'v']]
            choice = random.choice([0, 1, 2], 1)[0]
            directions = choices[choice]
        else:
            directions = [self.direction]

        for direction in directions:
            if direction == "v":
                data, rotation = self.flip_vertical(data, rotation)
            elif direction == "h":
                data, rotation = self.flip_horizontal(data, rotation)
            else:
                raise RuntimeError(f"Unknown direction: {direction}")

        # update params depends on lidar2camera and camera_intrinsics
        if "lidar2camera" in data:
            params = []
            for c2l in data['camera2lidar']:
                c2l = c2l.copy()
                _rot = c2l[:3, :3]
                _trans = c2l[:3, 3]
                l2c = np.eye(4)
                l2c[:3, :3] = _rot.T
                l2c[:3, 3] = -_rot.T @ _trans
                params.append(l2c)
            data['lidar2camera'] = params
        if "lidar2image" in data:
            params = []
            for l2c, cam_i in zip(
                    data['lidar2camera'], data['camera_intrinsics']):
                l2c = l2c.copy()
                cam_i = cam_i.copy()
                lidar2camera_r = l2c[:3, :3]
                lidar2camera_t = l2c[:3, 3]
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = lidar2camera_t
                lidar2image = cam_i @ lidar2camera_rt.T
                params.append(lidar2image)
            data['lidar2image'] = params
        if "camera2ego" in data:
            # I don't know how to handle this, just drop.
            data.pop("camera2ego")

        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data

# ==================== copy from mmdet3d ===================#

@PIPELINES.register_module()
class LoadMultiViewImageFromFiles:
    """Load multi channel images from a list of separate channel files.

    Expects results['image_paths'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["image_paths"]
        # img is of shape (h, w, c, num_views)
        # modified for waymo
        images = []
        h, w = 0, 0
        for name in filename:
            images.append(Image.open(name))
        
        #TODO: consider image padding in waymo

        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = images
        # [1600, 900]
        results["img_shape"] = images[0].size
        results["ori_shape"] = images[0].size
        # Set initial values for default meta_keys
        results["pad_shape"] = images[0].size
        results["scale_factor"] = 1.0
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
    """

    def __init__(
        self,
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_bbox=False,
        with_label=False,
        with_mask=False,
        with_seg=False,
        with_bbox_depth=False,
        poly2mask=True,
    ):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
        )
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results["gt_bboxes_3d"] = results["ann_info"]["gt_bboxes_3d"]
        results["bbox3d_fields"].append("gt_bboxes_3d")
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results["centers2d"] = results["ann_info"]["centers2d"]
        results["depths"] = results["ann_info"]["depths"]
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["gt_labels_3d"] = results["ann_info"]["gt_labels_3d"]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results["attr_labels"] = results["ann_info"]["attr_labels"]
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)

        return results


@PIPELINES.register_module()
class ImageAug3D:
    def __init__(
        self, final_dim, resize_lim, bot_pct_lim, rot_lim, rand_flip, is_train
    ):
        self.final_dim = final_dim
        self.resize_lim = resize_lim
        self.bot_pct_lim = bot_pct_lim
        self.rand_flip = rand_flip
        self.rot_lim = rot_lim
        self.is_train = is_train

    def sample_augmentation(self, results):
        W, H = results["ori_shape"]
        fH, fW = self.final_dim
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.bot_pct_lim)) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.rand_flip and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = np.mean(self.resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.bot_pct_lim)) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def img_transform(
        self, img, rotation, translation, resize, resize_dims, crop, flip, rotate
    ):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        imgs = data["img"]
        new_imgs = []
        transforms = []
        for img in imgs:
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            new_img, rotation, translation = self.img_transform(
                img,
                post_rot,
                post_tran,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            new_imgs.append(new_img)
            transforms.append(transform.numpy())
        data["img"] = new_imgs
        # update the calibration matrices
        data["img_aug_matrix"] = transforms
        return data


@PIPELINES.register_module()
class GlobalRotScaleTrans:
    def __init__(self, resize_lim, rot_lim, trans_lim, is_train):
        self.resize_lim = resize_lim
        self.rot_lim = rot_lim
        self.trans_lim = trans_lim
        self.is_train = is_train

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        transform = np.eye(4).astype(np.float32)

        if self.is_train:
            scale = random.uniform(*self.resize_lim)
            theta = random.uniform(*self.rot_lim)
            translation = np.array([random.normal(0, self.trans_lim) for i in range(3)])
            rotation = np.eye(3)

            if "points" in data:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        return data


@PIPELINES.register_module()
class ImageNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["img"] = [self.compose(img) for img in data["img"]]
        data["img_norm_cfg"] = dict(mean=self.mean, std=self.std)
        return data


@PIPELINES.register_module()
class DefaultFormatBundle3D:
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(
        self,
        classes,
        with_gt: bool = True,
        with_label: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = classes
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if "points" in results:
            assert isinstance(results["points"], BasePoints)
            results["points"] = DC(results["points"].tensor)

        for key in ["voxels", "coors", "voxel_centers", "num_points"]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if "gt_bboxes_3d_mask" in results:
                gt_bboxes_3d_mask = results["gt_bboxes_3d_mask"]
                results["gt_bboxes_3d"] = results["gt_bboxes_3d"][gt_bboxes_3d_mask]
                if "gt_names_3d" in results:
                    results["gt_names_3d"] = results["gt_names_3d"][gt_bboxes_3d_mask]
                if "centers2d" in results:
                    results["centers2d"] = results["centers2d"][gt_bboxes_3d_mask]
                if "depths" in results:
                    results["depths"] = results["depths"][gt_bboxes_3d_mask]
            if "gt_bboxes_mask" in results:
                gt_bboxes_mask = results["gt_bboxes_mask"]
                if "gt_bboxes" in results:
                    results["gt_bboxes"] = results["gt_bboxes"][gt_bboxes_mask]
                results["gt_names"] = results["gt_names"][gt_bboxes_mask]
            if self.with_label:
                if "gt_names" in results and len(results["gt_names"]) == 0:
                    results["gt_labels"] = np.array([], dtype=np.int64)
                    results["attr_labels"] = np.array([], dtype=np.int64)
                elif "gt_names" in results and isinstance(results["gt_names"][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results["gt_labels"] = [
                        np.array(
                            [self.class_names.index(n) for n in res], dtype=np.int64
                        )
                        for res in results["gt_names"]
                    ]
                elif "gt_names" in results:
                    results["gt_labels"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names"]],
                        dtype=np.int64,
                    )
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if "gt_names_3d" in results:
                    results["gt_labels_3d"] = np.array(
                        [self.class_names.index(n) for n in results["gt_names_3d"]],
                        dtype=np.int64,
                    )
        if "img" in results:
            results["img"] = DC(torch.stack(results["img"]), stack=True)

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if "gt_bboxes_3d" in results:
            if isinstance(results["gt_bboxes_3d"], BaseInstance3DBoxes):
                results["gt_bboxes_3d"] = DC(results["gt_bboxes_3d"], cpu_only=True)
            else:
                results["gt_bboxes_3d"] = DC(to_tensor(results["gt_bboxes_3d"]))
        return results


@PIPELINES.register_module()
class Collect3D:
    def __init__(
        self,
        keys,
        meta_keys=(
            "camera_intrinsics",
            "camera2ego",
            "img_aug_matrix",
            "lidar_aug_matrix",
        ),
        meta_lis_keys=(
            "filename",
            "timestamp",
            "ori_shape",
            "img_shape",
            "lidar2image",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "token",
            "pcd_scale_factor",
            "pcd_rotation",
            "lidar_path",
            "transformation_3d_flow",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys
        # [fixme] note: need at least 1 meta lis key to perform training.
        self.meta_lis_keys = meta_lis_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``metas``
        """
        data = {}
        for key in self.keys:
            if key not in self.meta_keys:
                data[key] = results[key]
        for key in self.meta_keys:
            if key in results:
                val = np.array(results[key])
                if isinstance(results[key], list):
                    data[key] = DC(to_tensor(val), stack=True)
                else:
                    data[key] = DC(to_tensor(val), stack=True, pad_dims=1)

        metas = {}
        for key in self.meta_lis_keys:
            if key in results:
                metas[key] = results[key]

        data["metas"] = DC(metas, cpu_only=True)
        return data
