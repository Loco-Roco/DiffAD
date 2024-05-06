# DiffAD
[ICCV2023] Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model

```
@inproceedings{zhang2023unsupervised,
  title={Unsupervised Surface Anomaly Detection with Diffusion Probabilistic Model},
  author={Zhang, Xinyi and Li, Naiqi and Li, Jiawei and Dai, Tao and Jiang, Yong and Xia, Shu-Tao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6782--6791},
  year={2023}
}
```

## Method overview
<img width="80%" alt="image" src="https://github.com/Loco-Roco/DiffAD/assets/51684540/789dd35c-17d6-48d5-8c57-612bc71d0d6e">

## Installation
```
conda env create -f environment.yaml
conda activate DiffAD
```

## Dataset
Following DRAEM, we use the MVTec-AD and DTD dataset. You can run the download_dataset.sh script from the project directory to download the MVTec and the DTD datasets to the datasets folder in the project directory:
```
./scripts/download_dataset.sh
```

## Training
### Reconstruction sub-network
The reconstrucion sub-network is based on the latent diffusion model. 
#### Training Auto-encoder
```
cd rec_network
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/kl.yaml -t --gpus 0,  
```
#### Training LDMs
```
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/mvtec.yaml -t --gpus 0 -max_epochs 4000, 
```

### Discriminative sub-network
```
cd seg_network
CUDA_VISIBLE_DEVICES=<GPU_ID> python train.py --gpu_id 0 --lr 0.001 --bs 32 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/obj_name --log_path ./logs/
```

## Evaluating
### Reconstrucion performance
After training the reconstruction sub-network, you can test the reconstruction performance with the anomalous inputs:
```
python scripts/mvtec.py
```
For some samples with severe deformations, such as missing transistors, you can add some noise to the anomalous conditions to adjust the sampling. 

### Anomaly segmentation
```
cd seg_network
python test.py --gpu_id 0 --base_model_name "seg_network" --data_path ./datasets/mvtec/ --checkpoint_path ./checkpoints/obj_name/
```


