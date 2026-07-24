# DRG-Net

DRG-Net is a rotation-aware geometric-consistency framework for robust point cloud registration. The method is developed based on the PARE-Net-style registration pipeline and introduces three geometric enhancement modules: Dual-Stream Geometric Encoder (DSGE), Dual Rotation-Invariant Positional Encoding (DRIPE), and Point-wise Geometric Feature Enhancement (PGFE).

The goal of DRG-Net is not to simply increase the number of correspondences, but to improve the geometric quality and transformation usefulness of the retained correspondences.

**Project repository:**  
https://github.com/ILCG-wyp/DRGNet

---


🚀 Installation

We recommend using a conda environment.

```bash
conda create -n drgnet python=3.8
conda activate drgnet
Install PyTorch with CUDA 11.6:

bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
Install other dependencies:

bash
pip install -r requirements.txt
python setup.py build develop
If the project contains custom point operations, please compile them:

bash
cd drgnet/extensions/pointops/
python setup.py install
cd ../../../

💻 Tested Environment
The code has been tested under the following environment:

Item	Version
OS	Ubuntu 20.04
Python	3.8
PyTorch	1.13.0
CUDA	11.6
cuDNN	8.0.5
GCC	7.5.0
GPU	NVIDIA RTX 3090
CPU	Intel Xeon Silver 4314
Batch size for profiling	1
Runtime and memory may vary slightly across different devices.

🗂️ Dataset Preparation
DRG-Net uses publicly available datasets, including 3DMatch, 3DLoMatch, and KITTI. No new raw dataset is generated in this work. The rotated benchmarks are generated from the original test sets following the rotation protocols described in the paper.

📍 3.1 3DMatch / 3DLoMatch
Please download 3DMatch from the official dataset page (http://3dmatch.cs.princeton.edu/) or prepare the processed 3DMatch data following the standard setting used in GeoTransformer / PARE-Net-style evaluation.

Expected directory structure:

text
data/3DMatch/
├── train/
│   ├── 7-scenes-chess/
│   │   ├── cloud_bin_0.pth
│   │   └── ...
│   └── ...
├── test/
│   ├── 7-scenes-redkitchen/
│   │   ├── cloud_bin_0.pth
│   │   └── ...
│   └── ...
├── train_pair_overlap_masks/
│   └── ...
└── metadata/
    └── ...
Please modify the dataset path in:

text
experiments/3DMatch/config.py
For example:

python
_C.data.dataset_root = "/your/path/to/3DMatch"
3DLoMatch uses the same processed 3DMatch data and a different benchmark split during evaluation.

📍 3.2 KITTI
Please download KITTI from the official dataset page (http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and organize the processed data as follows:

text
data/KITTI/
├── downsampled/
│   ├── 00/
│   │   ├── 000000.npy
│   │   └── ...
│   └── ...
├── train_pair_overlap_masks/
│   └── ...
└── metadata/
    └── ...
Please modify the KITTI dataset path in the corresponding configuration file:

python
_C.data.dataset_root = "/your/path/to/KITTI"

📍 3.3 Rotated Benchmarks
The rotated variants are generated from the original test pairs. In the full SO(3) protocol, rotations are applied to point clouds and the ground-truth transformations are updated accordingly. Normals are rotated using the same rotation matrices to keep the PPF-based geometric encoding consistent.

For evaluation, the rotated benchmark can be enabled by the --rotated flag.

Rotated 3DMatch

bash
python test.py --benchmark 3DMatch --rotated --snapshot xxx.pth.tar
python eval.py --benchmark 3DMatch
Rotated 3DLoMatch

bash
python test.py --benchmark 3DLoMatch --rotated --snapshot xxx.pth.tar
python eval.py --benchmark 3DLoMatch

🏋️ Training

🎯 4.1 Train on 3DMatch
bash
CUDA_VISIBLE_DEVICES=0 python trainval.py
If you use the experiment folder structure, run:

bash
cd experiments/3DMatch
CUDA_VISIBLE_DEVICES=0 python trainval.py

🎯 4.2 Train on KITTI
bash
cd experiments/KITTI
CUDA_VISIBLE_DEVICES=0 python trainval.py
The trained checkpoints are saved by default to:

text
output/3DMatch/features/snapshots/epoch-xxx.pth.tar
Please replace epoch-xxx.pth.tar with the actual checkpoint file.

🎯 4.3 Random Seeds
The main reported model uses seed 7351. For the three-seed stability analysis, the following seeds are used:

Purpose	Seeds
Main experiment	7351
Three-seed stability	7351, 1234, 2023
Please modify the seed in the corresponding configuration file or command-line argument according to your implementation.

🧪 Testing
✅ 5.1 3DMatch
bash
python test.py --benchmark 3DMatch --snapshot xxx.pth.tar
python eval.py --benchmark 3DMatch
✅ 5.2 3DLoMatch
bash
python test.py --benchmark 3DLoMatch --snapshot xxx.pth.tar
python eval.py --benchmark 3DLoMatch
✅ 5.3 Rotated 3DMatch
bash
python test.py --benchmark 3DMatch --rotated --snapshot xxx.pth.tar
python eval.py --benchmark 3DMatch
✅ 5.4 Rotated 3DLoMatch
bash
python test.py --benchmark 3DLoMatch --rotated --snapshot xxx.pth.tar
python eval.py --benchmark 3DLoMatch
✅ 5.5 KITTI
bash
python test.py --benchmark KITTI --snapshot xxx.pth.tar
python eval.py --benchmark KITTI
If your KITTI script does not use the --benchmark argument, please run:

bash
python test.py --snapshot xxx.pth.tar
python eval.py

📊 Expected Logs and Results
Expected logs are provided in:

text
logs/expected/
├── 3dmatch_eval.log
├── 3dlomatch_eval.log
├── rotated_3dmatch_eval.log
├── rotated_3dlomatch_eval.log
└── kitti_eval.log
Expected result files are provided in:

text
results/expected/
├── table1_3dmatch.csv
├── table_rotated_benchmarks.csv
├── table_topk.csv
├── table_normal_robustness.csv
└── table_efficiency.csv

📌 Notes on Checkpoints
This repository does not host pretrained checkpoints. To evaluate DRG-Net, please train the model following the training instructions above and specify the generated checkpoint using the --snapshot argument.

Example:

bash
python test.py --benchmark 3DMatch --snapshot output/3DMatch/features/snapshots/epoch-xxx.pth.tar

🙏 Acknowledgements
This project is developed based on the point cloud registration pipeline of PARE-Net. We also thank the authors of GeoTransformer, VectorNeurons, PAConv, RoITr, and related open-source projects.

📄 License
This repository is released for academic research. Please check the LICENSE file for details.

