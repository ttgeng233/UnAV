# Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline

[[Project page]](https://unav100.github.io/) [[ArXiv]](https://arxiv.org/abs/2303.12930v2) [[Dataset(Hugging face)]](https://huggingface.co/datasets/ttgeng233/UnAV-100) [[Dataset(Google drive)]](https://drive.google.com/drive/folders/1X4eoCPPtqi0_IKd2JGOv_q383xd27Ybb) [[Dataset(Baidu drive)]](https://pan.baidu.com/share/init?surl=en_Ni7X8-zKwqbUG7ITXig&pwd=6c48) [[Benchmark]](https://paperswithcode.com/sota/audio-visual-event-localization-on-unav-100)

This repository contains code for CVPR 2023 paper "[Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline](https://openaccess.thecvf.com/content/CVPR2023/html/Geng_Dense-Localizing_Audio-Visual_Events_in_Untrimmed_Videos_A_Large-Scale_Benchmark_and_CVPR_2023_paper.html)". This paper introduces the first Untrimmed Audio-Visual (UnAV-100) dataset and proposes to sovle audio-visual event localization problem in more realistic and challenging scenarios. 


## Requirements
The implemetation is based on PyTorch. Follow [INSTALL.md](INSTALL.md) to install required dependencies.

## Data preparation
The proposed UnAV-100 dataset can be downloaded from [[Project Page]](https://unav100.github.io/), including YouTube links of raw videos, annotations and extracted features. 

If you want to use your own choices of video features, you can download the raw videos from this [link](https://pan.baidu.com/s/1N2bNc288vK9PDpHkrPBx2A) (Baidu Drive, pwd: qslx). A download script is also provided for raw videos at `scripts/video_download.py`. 

**Note**: after downloading data, unpack files under `data/unav100`. The folder structure should look like:
```
This folder
│   README.md
│   ...  
└───data/
│    └───unav100/
│    	 └───annotations/
│               └───unav100_annotations.json
│    	 └───av_features/   
│               └───__2MwJ2uHu0_flow.npy    # mix all features together
│               └───__2MwJ2uHu0_rgb.npy 
│               └───__2MwJ2uHu0_vggish.npy 
|                   ...
└───libs
│   ...
```
## Training
Run ```train.py``` to train the model on UnAV-100 dataset. This will create an experiment folder under ```./ckpt``` that stores training config, logs, and checkpoints.
```
python ./train.py ./configs/avel_unav100.yaml --output reproduce
```

## Evaluation
Run ```eval.py``` to evaluate the trained model. 
```
python ./eval.py ./configs/avel_unav100.yaml ./ckpt/avel_unav100_reproduce
```
[Optional] We also provide a pretrained model for UnAV-100, which can be downloaded from [this link](https://drive.google.com/file/d/1qiC2osEaBSH8HFvF0WY_535F21CM3JXj/view?usp=share_link).

## Citation
If you find our dataset and code are useful for your research, please cite our paper
```
@inproceedings{geng2023dense,
  title={Dense-Localizing Audio-Visual Events in Untrimmed Videos: A Large-Scale Benchmark and Baseline},
  author={Geng, Tiantian and Wang, Teng and Duan, Jinming and Cong, Runmin and Zheng, Feng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22942--22951},
  year={2023}
}
```

## Acknowledgement
The video features of I3D-rgb & flow and Vggish-audio were extracted using [video_features](https://github.com/v-iashin/video_features). Our baseline model was implemented based on [ActionFormer](https://github.com/happyharrycn/actionformer_release). We thank the authors for sharing their codes. If you use our code, please consider to cite their works.
