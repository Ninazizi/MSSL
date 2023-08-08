The codes for "Generalized representations of microstructural images 
using self-supervised learning" is presented here.  

The codes are developed from the MMSelfSup framework.

To fully reproduce the paper, you can follow the steps.

1. Install the environments according to README.md
2. Download the datasets, unzip and put it under mmselfsup-0.9-MSSL/
3. Train the self-supervised learning model with the unlabeled dataset.
```shell
sh tools/dist_train.sh configs/selfsup/simsiam/simsiam_resnet18_8xb32-coslr-400e_sem.py 0
```
The training output is also provided in work_dirs/selfsup/simsiam_resnet18_8xb32-coslr-400e_sem
After the whole training process, you will get the epoch_400.pth.
4. Convert the trained epoch_400.pth to the pretrained.pth
```shell
python tools/model_converters/extract_backbone_weights.py work_dirs/selfsup/simsiam_resnet18_8xb32-coslr-400e_sem/epoch_400.pth work_dirs/selfsup/simsiam_resnet18_8xb32-coslr-400e_sem/pretrained.pth
```
5. Transfer learning (task 1-3) with the pretrained.pth
```shell
sh tools/benchmarks/classification/dist_train_linear.sh configs/benchmarks/classification/steel/resnet18_linear-8xb32-coslr-100e_task1.py work_dirs/selfsup/simsiam_resnet18_8xb32-coslr-400e_sem/pretrained.pth
```
6. Quantify the homogeneity. Put microstructure images under folder data_collect and run
```shell
python tools/benchmarks/classification/knn_imagenet/test_homogeneity.py
```