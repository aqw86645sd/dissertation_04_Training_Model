# dissertation_04_Training_Model1
論文 - 訓練 model 1 & 2 並儲存 model and weights (python)


# 環境
M1版本 pytorch pip安裝會有問題

所以要用conda安裝(目前用3.7.9沒有問題)


# conda command
conda create --name conda_venv_37_pytorch python=3.7

conda activate conda_venv_37_pytorch

conda install torch

conda install -c anaconda scikit-learn


# 設定 interpreter to existed conda


# 問題處理方法
### OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized. #1715
增加環境變數，並重開機

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'



.

.

.
