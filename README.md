# DRGNet
point cloud registration

1.Installation

We recommend using a conda environment for installation.

# Create and activate a new environment
conda create -n drgnet python=3.8
conda activate drgnet

# Install PyTorch 1.13.0 with CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

The code has been tested with Ubuntu 20.04, GCC 7.5.0, Python 3.8, PyTorch 1.13.0, CUDA 11.6, and cuDNN 8.0.5.

2.Dataset

3DMatch and KITTI datasets can be downloaded from Google Drive or Baidu Disk (extraction code: qyhn).

3DMatch should be organized as follows:
--your_3DMatch_path--3DMatch
              |--train--7-scenes-chess--cloud_bin_0.pth
                    |--     |--...         |--...
              |--test--7-scenes-redkitchen--cloud_bin_0.pth
                    |--     |--...         |--...
              |--train_pair_overlap_masks--7-scenes-chess--masks_1_0.npz
                    |--     |--...         |--...       


KITTI should be organized as follows:
--your_KITTI_path--KITTI
            |--downsampled--00--000000.npy
                    |--...   |--... |--...
            |--train_pair_overlap_masks--0--masks_11_0.npz
                    |--...   |--... |--...


3.Training

You can train DRG-Net on 3DMatch or KITTI using the commands below.

Single GPU

cd experiments/3DMatch        # or KITTI
CUDA_VISIBLE_DEVICES=0 python trainval.py

Multiple GPUs

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 trainval.py


4.Testing

3DMatch：

python test.py --benchmark 3DMatch
python eval.py --benchmark 3DMatch

For 3DLoMatch, change --benchmark to 3DLoMatch.


KITTI：

python test.py
python eval.py
