# HPLFlowNet
This is the code for [HPLFlowNet](https://web.cs.ucdavis.edu/~yjlee/projects/cvpr2019-HPLFlowNet.pdf), a hierarchical permutohedral lattice flownet for scene flow estimation on large-scale point clouds. The code is developed and maintained by Xiuye Gu.

## Prerequisites
Our model is trained and tested under:
* Python 3.5.2 (testing under Python 3.6.5 also works)
* NVIDIA GPU + CUDA CuDNN
* PyTorch (torch == 0.4.0)
* Numba (numba == 0.38.1)
* You may need to install cffi.
* Mayavi for visualization. 

* Installation on Ubuntu:
```bash
pip3 install https://download.pytorch.org/whl/cu90/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip3 install numba
pip3 install cffi
sudo apt-get -y install python-vtk
sudo pip3 install mayavi
sudo apt-get install python3-pyqt5
sudo pip3 install PyQt5
pip3 install imageio
pip3 install tensorboard
```

## Data preprocess

### FlyingThings3D
Download and unzip the "Disparity", "Disparity Occlusions", "Disparity change", "Optical flow", "Flow Occlusions" for DispNet/FlowNet2.0 dataset subsets from the [FlyingThings3D website](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (we used the paths from [this file](https://lmb.informatik.uni-freiburg.de/data/FlyingThings3D_subset/FlyingThings3D_subset_all_download_paths.txt), now they added torrent downloads)
. They will be upzipped into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_flyingthings3d_subset.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/FlyingThings3D_subset_processed_35m --only_save_near_pts
```

### KITTI Scene Flow 2015
Download and unzip [KITTI Scene Flow Evaluation 2015](http://www.cvlibs.net/download.php?file=data_scene_flow.zip) to directory `RAW_DATA_PATH`.
Run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_kitti.py RAW_DATA_PATH SAVE_PATH/KITTI_processed_occ_final
```

### RefRESH
Download ZIPs of all scenes from [RefRESH Google doc](https://drive.google.com/drive/folders/1Im1_ehSg4ALzeGctYGzvUv9KhcRlHXu_).
Unzip all of the scenes into the same directory, `RAW_DATA_PATH`. Then run the following script for 3D reconstruction:

```bash
python3 data_preprocess/process_refresh_rigidity.py --raw_data_path RAW_DATA_PATH --save_path SAVE_PATH/REFRESH_pc --only_save_near_pts
```

## Get started
Setup:
```bash
cd models; python3 build_khash_cffi.py; cd ..
```

### Train
Set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section. Then run
```bash
python3 main.py configs/train_xxx.yaml
```

### Test
Set `data_root` in the configuration file to `SAVE_PATH` in the data preprocess section. Set `resume` to be the path of your trained model or our trained model in `trained_models`. Then run
```bash
python3 main.py configs/test_xxx.yaml
```

Current implementation only supports `batch_size=1`.

### Visualization
If you set `TOTAL_NUM_SAMPLES` in `evaluation_bnn.py` to be larger than 0. Sampled results will be saved in a subdir of your checkpoint directory, `VISU_DIR`.

Run
```bash
python3 visualization.py VISU_DIR
``` 

## Citation

If you use this code for your research, please cite our paper.


```
@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
```
## Acknowledgments
Our permutohedral lattice implementation is based on [Fast High-Dimensional Filtering Using the Permutohedral Lattice](http://graphics.stanford.edu/papers/permutohedral/). The [BilateralNN](https://github.com/MPI-IS/bilateralNN) implementation is also closely related.
Our hash table implementation is from [khash-based hashmap in Numba](https://github.com/synapticarbors/khash_numba).
