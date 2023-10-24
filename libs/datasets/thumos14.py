import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import torchaudio

from .datasets import register_dataset, make_generator
from .data_utils import truncate_feats
from torchaudio.transforms import MelSpectrogram

from scipy.interpolate import interp1d
import librosa


@register_dataset("thumos_audio")
class THUMOS14AudioDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        audio_folder,
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        max_buffer_len_factor,
        backbone_arch,
        regression_range,
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        audio_format,
        class_aware,
        scale_factor,
        
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        self.audio_folder = audio_folder
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
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.audio_format = audio_format

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.3, 0.7, 5),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }
        
        
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        self.reg_range = regression_range
        self.class_aware = class_aware
         
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len' : self.max_seq_len * max_buffer_len_factor,
                'fpn_levels' : len(self.fpn_strides),
                'scale_factor' : scale_factor,
                'regression_range' : self.reg_range
            }
        )
        
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
            # or does not have the feature file
            feat_file = os.path.join(self.feat_folder,
                                     self.file_prefix + key + self.file_ext)
            if not os.path.exists(feat_file):
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
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # our code can now handle this corner case
                segments, labels = [], []
                for act in value['annotations']:
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])

                segments = np.asarray(segments, dtype=np.float32)
                labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
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

    def __len__(self):
        return len(self.data_list)

    
    def __getitem__(self, idx):
        video_item = self.data_list[idx]

        # Load video features
        filename = os.path.join(self.feat_folder, self.file_prefix + video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)
        feats = torch.from_numpy(np.ascontiguousarray(feats))
        feats = feats.transpose(1, 0)
        
        if self.audio_format == "raw_wav":    
            audio_file = os.path.join(self.raw_audio_folder, self.file_prefix + video_item['id'] + ".wav")
            raw_audio_data, sample_rate = torchaudio.load(audio_file, normalize=True)
            raw_audio_data = raw_audio_data.mean(dim=0, keepdim=True)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(raw_audio_data)
            waveform = (waveform - waveform.mean()) / waveform.std()

            # Cut into 1-second (16000 samples) chunks
            num_segments = waveform.shape[1] // 2133
            segments = []
            for i in range(num_segments):
                segment = waveform[:, i*2133:(i+1)*2133]
                segments.append(segment)
            raw_audio_data = torch.stack(segments).squeeze().transpose(0, 1)
        elif self.audio_format == "mel_spec": 
            audio_file = os.path.join(self.audio_folder, self.file_prefix + video_item['id'] + self.file_ext) 
            audio_feats = np.load(audio_file).astype(np.float32)
        elif self.audio_format == "vgg": 
            audio_file = os.path.join(self.audio_folder, self.file_prefix + video_item['id'] + self.file_ext) 
            audio_feats = np.load(audio_file).astype(np.float32) / 255
        else:
            raise Exception(f"{self.audio_format} not recognized")
            
        audio_feats = torch.from_numpy(np.ascontiguousarray(audio_feats)).transpose(0, 1) 
        
        resize_audio_feats = F.interpolate(
            audio_feats.unsqueeze(0),
            size=feats.size(1),
            mode='linear',
            align_corners=False
        )
        audio_feats = resize_audio_feats.squeeze(0)
            
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        if video_item['segments'] is not None:
            segments = torch.from_numpy(video_item['segments'] * video_item['fps'] / feat_stride - feat_offset)
            labels = torch.from_numpy(video_item['labels'])
        else:
            segments, labels = None, None
            
        feats = {'visual': feats, 'audio': audio_feats}

        # Prepare the data dictionary
        data_dict = {
            'video_id': video_item['id'],
            'feats': feats,      # C x T
            'segments': segments,         # N x 2
            'labels': labels,             # N
            'fps': video_item['fps'],
            'duration': video_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': self.num_frames
        }

        # Truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio)
                # compute the gt labels for cls & reg
        gt_segments = data_dict['segments']
        gt_labels = data_dict['labels']
        points = self.point_generator(self.fpn_strides)
        data_dict['gt_cls_labels'], data_dict['gt_offsets'] = self.label_points(
                points, gt_segments, gt_labels)
        data_dict['points'] = points
        
        return data_dict
