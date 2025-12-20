# UNC-Africa Lymphoma MIL foundational model training code

# data preprocessing
Please refer to Trident (https://github.com/mahmoodlab/TRIDENT) to create patches and feature embedding for each foundataional model and store them as h5 files

# train
use configs/config_MIL_africa_lymphoma.json for basic file path settings and training parameters setting.
run train_cv.py for training and get cross-validated results
