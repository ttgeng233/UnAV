import yaml


DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891, 
    # dataset loader, specify the dataset here
    "dataset_name": "unav100",
    "devices": ['cuda:0'], # default: single gpu
    "train_split": ('train', ),
    "val_split": ('validation', ),
    "test_split": ('test', ),
    "model_name": "LocPointTransformer",
    "dataset": {
        # temporal stride of the feats
        "feat_stride": 8,
        # number of frames for each feat
        "num_frames": 24,
        "default_fps": 25,
        # number of classes
        "num_classes": 100,
        # downsampling rate of features, 1 to use original resolution
        "downsample_rate": 1,
        # max sequence length during training
        "max_seq_len": 224,
        # max buffer size (defined a factor of max_seq_len)
        "max_buffer_len_factor": 1.0, 
        # threshold for truncating an action
        "trunc_thresh": 0.5,
        # set to a tuple (e.g., (0.9, 1.0)) to enable random feature cropping
        "crop_ratio": [0.9, 1.0],
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        "backbone_type": 'convTransformer',
        "dependency_type": "DependencyBlock",
        "backbone_arch": (2, 3, 5),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # regression range for pyramid levels
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # number of heads in self-attention
        "n_head": 4,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
    },
    "train_cfg": {
        # on reg_loss, use -1 to enable auto balancing
        "loss_weight": -1, 
        # use prior in model initialization to improve stability
        "cls_prior_prob": 0.01,
        "init_loss_norm": 250,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": 1.0,
        # cls head without data 
        "head_empty_cls": [],  
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft', # soft | hard | none
        "nms_sigma" : 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh" : 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # fill in derived fields
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["dataset"]["backbone_arch"] = config["model"]["backbone_arch"]
    config["dataset"]["regression_range"] = config["model"]["regression_range"]
    config["dataset"]["class_aware"] = config["model"]["class_aware"]
    config["dataset"]["scale_factor"] = config["model"]["scale_factor"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config
