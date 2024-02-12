import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import nn

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats

@register_dataset("unav100")
class UnAV100Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        max_buffer_len_factor, # max buffer size (defined a factor of max_seq_len)
        scale_factor,          # scale factor between branch layers
        regression_range,      # regression range on each level of FPN
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        class_aware,            # if use class-aware regression
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        # "empty" noun categories on epic-kitchens
        assert len(label_dict) <= num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        empty_label_ids = self.find_empty_cls(label_dict, num_classes)
        self.db_attributes = {
            'dataset_name': 'unav-100',
            'tiou_thresholds': np.linspace(0.1, 0.9, 9), 
            'empty_label_ids': empty_label_ids
        }

        # location generator: points
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        self.reg_range = regression_range
        self.class_aware = class_aware

        max_div_factor = 1
        for l, stride in enumerate(self.fpn_strides):
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len_ori' : self.max_seq_len,
                'max_buffer_len_factor': max_buffer_len_factor, #########
                'fpn_levels' : len(self.fpn_strides),
                'scale_factor' : scale_factor,
                'regression_range' : self.reg_range,
                'max_div_factor' : self.max_div_factor  #########
            }
        )


    def find_empty_cls(self, label_dict, num_classes):
        # find categories with out a data sample
        if len(label_dict) == num_classes:
            return []
        empty_label_ids = []
        label_ids = [v for _, v in label_dict.items()]
        for id in range(num_classes):
            if id not in label_ids:
                empty_label_ids.append(id)
        return empty_label_ids

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                num_acts = len(value['annotations'])
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(value['annotations']):
                    segments[idx][0] = act['segment'][0]
                    segments[idx][1] = act['segment'][1]
                    labels[idx] = label_dict[act['label']]
            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def label_points(self, points, gt_segments, gt_labels):
        # concat points on all pyramid levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        concat_points = torch.cat(points, dim=0)
        cls_targets, reg_targets = self.label_points_single_video(
            concat_points, gt_segments, gt_labels)
        return cls_targets, reg_targets
    
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        # concat_points : F T x 4 (t, regressoin range, stride)
        # gt_segment : N (#Events) x 2
        # gt_label : N (#Events) x 1
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0] 
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        # inside an gt action
        inside_gt_seg_mask = reg_targets.min(-1)[0] > 0 

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        lens.masked_fill_(inside_gt_seg_mask==0, float('inf')) 
        lens.masked_fill_(inside_regress_range==0, float('inf'))

        if self.class_aware:
            min_len  = lens
            min_len_mask = (min_len < float('inf')).to(reg_targets.dtype) 
        else:
            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)
            # corner case: multiple actions with very similar durations 
            min_len_mask = torch.logical_and(
                (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
            ).to(reg_targets.dtype) 

        # cls_targets: F T x C; reg_targets F T x 2
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        cls_targets = min_len_mask @ gt_label_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        
        if self.class_aware:
            new_reg_targets = torch.zeros(num_pts, self.num_classes, 2)
            for i in range(num_pts):
                inds = min_len_mask[i].nonzero() 
                new_reg_targets[i, gt_label[inds]] = reg_targets[i, inds] 
            new_reg_targets /= concat_points[:, 3, None, None]
        else:        
            # OK to use min_len_inds
            new_reg_targets = reg_targets[range(num_pts), min_len_inds] 
            # normalization based on stride
            new_reg_targets /= concat_points[:, 3, None]

        return cls_targets, new_reg_targets

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point 
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load visual features (rgb and optical flow)
        filename_rgb = os.path.join(self.feat_folder,
                        self.file_prefix + video_item['id'] + '_rgb' + self.file_ext)
        feats_rgb = np.load(filename_rgb).astype(np.float32)
        filename_flow = os.path.join(self.feat_folder,
                        self.file_prefix + video_item['id'] + '_flow' + self.file_ext)                                                       
        feats_flow = np.load(filename_flow).astype(np.float32)
        # concat rgb and flow
        feats_visual = np.hstack((feats_rgb, feats_flow)) 
        # deal with downsampling (= increased feat stride)
        feats_visual = feats_visual[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate

        # T x C -> C x T
        feats_visual = torch.from_numpy(np.ascontiguousarray(feats_visual.transpose()))

        #load audio features
        filename_audio = os.path.join(self.feat_folder,
                            self.file_prefix + video_item['id'] + '_vggish' + self.file_ext)
        feats_audio = np.load(filename_audio).astype(np.float32)
        feats_audio = feats_audio[::self.downsample_rate, :]
        # T x C -> C x T
        feats_audio = torch.from_numpy(np.ascontiguousarray(feats_audio.transpose()))

        #avoid audio and visual features have different lengths
        min_len = min(feats_visual.shape[1], feats_audio.shape[1])
        feats = {'visual': feats_visual[:, :min_len], 'audio': feats_audio[:, :min_len] }
        
        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (video_item['segments'] * video_item['fps']- 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        # compute the gt labels for cls & reg
        gt_segments = data_dict['segments']
        gt_labels = data_dict['labels']
        points = self.point_generator(self.fpn_strides, feats['visual'], self.is_training)
        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = self.label_points(
                points, gt_segments, gt_labels)
        data_dict['points'] = points

        return data_dict
