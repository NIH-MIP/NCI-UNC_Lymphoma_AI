import argparse
import collections
import datetime
import gc
import os
import time
import warnings

import monai
import numpy as np
import scipy.stats as st
import torch
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_pool
from optim.optimizer import build_optimizer
from optim.scheduler import build_scheduler
from base import base_data_loader
from data_loader import dataset
from parse_config import ConfigParser
from trainer.trainer_MIL import Trainer
from utils import util

warnings.filterwarnings("ignore")


def main(pars, opts):
    # Fix random seeds for reproducibility
    config = ConfigParser.from_args(pars, opts)
    params = config.config
    SEED = pars.parse_args().seed
    util.seed_torch(SEED)
    torch.autograd.set_detect_anomaly(False)
    local_rank = pars.parse_args().device
    device = torch.device(f"cuda:{local_rank}")
    params['distributed'] = False
    logger = config.get_logger(name='train')
    logger.info("seed using is:" + str(SEED))
    
    # Filter and split up data
    patient_df = pd.read_csv(params['files']['data_sheet'])
    label_name = params['inputs']['label_name']

    patient_df = patient_df[patient_df[label_name].notna()]
    labels = np.array(patient_df[label_name])

    slide_label_counts = patient_df[label_name].value_counts()
    patient_label_counts = patient_df.drop_duplicates(subset="patient_id")[label_name].value_counts()
    # Ensure both sets of labels have the same columns
    all_labels = sorted(set(slide_label_counts.index).union(set(patient_label_counts.index)))
    slide_label_counts = slide_label_counts.reindex(all_labels, fill_value=0)
    patient_label_counts = patient_label_counts.reindex(all_labels, fill_value=0)
    
    logger.info("Total patients:")
    logger.info(patient_df['patient_id'].nunique())
    logger.info("Total slides: ")
    logger.info(len(patient_df['slide_id'].tolist()))
    if "VARIANT" not in label_name:
        # Combine results into a DataFrame, not good for multilabel display, when training for Variant, stop this
        result_df = pd.DataFrame({
            "Slide-Level Counts": slide_label_counts,
            "Patient-Level Counts": patient_label_counts
        }).T  # Transpose for horizontal display

        # Convert DataFrame to a formatted string
        formatted_output = result_df.to_string(index=True)
        logger.info("\n" + formatted_output)

    # control_group = patient_df['patient_id'] + patient_df['turbt_secondaryPath'].astype(str)
    control_group = patient_df['patient_id']

    # Split into cross-validated folds
    skgf = StratifiedGroupKFold(n_splits=params['trainer']['cv_fold'], shuffle=True, random_state=SEED)
        
    # Declare dictionaries of performance metrics
    best_auc = [0.] * int(params['trainer']['cv_fold'])
    best_acc = [0.] * int(params['trainer']['cv_fold'])
    best_prec = [0.] * int(params['trainer']['cv_fold'])
    best_sens = [0.] * int(params['trainer']['cv_fold'])
    best_spec = [0.] * int(params['trainer']['cv_fold'])
    best_f1 = [0.] * int(params['trainer']['cv_fold'])

    best_pid = [0.] * int(params['trainer']['cv_fold'])
    best_cutoff = [0.] * int(params['trainer']['cv_fold'])
    best_probability = [0.] * int(params['trainer']['cv_fold'])
    best_predicted = [0.] * int(params['trainer']['cv_fold'])
    best_target = [0.] * int(params['trainer']['cv_fold'])

    for fold, (train_index, valid_index) in enumerate(skgf.split(patient_df, labels, groups=control_group)):
        logger.info("training for fold {}".format(fold))
        df_train, df_valid = patient_df.iloc[train_index], patient_df.iloc[valid_index]

        if "MIL_Lymphoma" in params['name']:
            data_train = dataset.Lymphoma_Dataset_slide_lvl(params,df_train,device)
            data_valid = dataset.Lymphoma_Dataset_slide_lvl(params,df_valid,device)
        else:
            raise ValueError("Please check project name for config...")
        
        # Instantiate all data loaders
        train_loader_par = params['data_loader_train']['args']
        train_loader = base_data_loader.BaseDataLoader(data_train, batch_size=train_loader_par['batch_size'],
                                                       shuffle=train_loader_par['shuffle'],
                                                       num_workers=train_loader_par['num_workers'],
                                                       validation_split=train_loader_par['validation_split'], seed=SEED)
        valid_loader_par = params['data_loader_valid']['args']
        valid_loader = base_data_loader.BaseDataLoader(data_valid, batch_size=valid_loader_par['batch_size'],
                                                       shuffle=valid_loader_par['shuffle'],
                                                       num_workers=valid_loader_par['num_workers'],
                                                       validation_split=valid_loader_par['validation_split'], seed=SEED)

        # build model architecture, then print to console
        model = config.init_obj('arch', module_pool)
        model = model.to(device)
        # Compile model for PyTorch 2.0+ to speed up training
        # if torch.__version__ >= "2.0":
        #     model = torch.compile(model)

        # get function handles of loss and metrics
        criterion = getattr(module_loss, config['loss']['type'])
        metrics = [getattr(module_metric, met) for met in config['metrics']]

        # Build optim, learning rate scheduler. delete every line
        # containing lr_scheduler for disabling scheduler
        optimizer = build_optimizer(params, model)
        if params['trainer']['accumulate_steps'] > 1:
            lr_scheduler = build_scheduler(params, optimizer, len(train_loader)//params['trainer']['accumulate_steps'])
        else:
            lr_scheduler = build_scheduler(params, optimizer, len(train_loader))
        # lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        if params['trainer']['auto_resume']:
            model_path = os.path.join(config['trainer']['save_dir'], 'models', config['name'], params['run_id'])
            resume_file = util.auto_resume_helper(model_path, fold=fold)
            if resume_file:
                logger.info(f'auto resuming from {resume_file}')
                config.resume = resume_file
            else:
                logger.info(f'no checkpoint found in {resume_file}, ignoring auto resume')
                config.resume = None
        monai.data.set_track_meta(False)

        # Initialize trainer
        trainer = Trainer(model, criterion, metrics, optimizer,
                          config=config,
                          device=device,
                          data_loader=train_loader,
                          valid_data_loader=valid_loader,
                          dataset=None,
                          lr_scheduler=lr_scheduler,
                          cv=fold
                          )
        if params['train_mode'] == "train":
            trainer.train()
            gc.collect()
            torch.cuda.empty_cache()

        # Evaluate the model
        (best_auc[fold], best_prec[fold], best_sens[fold], best_spec[fold], best_acc[fold], best_f1[fold], best_pid[fold],
         best_cutoff[fold], best_probability[fold], best_predicted[fold],best_target[fold]) = trainer.evaluate(mode="valid")

    best_auc  = torch.tensor(best_auc,  dtype=torch.float32).cpu()
    best_prec = torch.tensor(best_prec, dtype=torch.float32).cpu()
    best_acc  = torch.tensor(best_acc,  dtype=torch.float32).cpu()
    best_sens = torch.tensor(best_sens, dtype=torch.float32).cpu()
    best_spec = torch.tensor(best_spec, dtype=torch.float32).cpu()
    best_f1   = torch.tensor(best_f1,   dtype=torch.float32).cpu()
    logger.info("ROC-AUC:     %.4f (%.4f)" % (np.mean(np.array(best_auc)), np.std(np.array(best_auc))))
    logger.info("Accuracy:    %.4f (%.4f)" % (np.mean(np.array(best_acc)), np.std(np.array(best_acc))))
    logger.info("Sensitivity: %.4f (%.4f)" % (np.mean(np.array(best_sens)), np.std(np.array(best_sens))))
    logger.info("Specificity: %.4f (%.4f)" % (np.mean(np.array(best_spec)), np.std(np.array(best_spec))))
    logger.info("Precision:   %.4f (%.4f)" % (np.mean(np.array(best_prec)), np.std(np.array(best_prec))))
    logger.info("F1 Score:   %.4f (%.4f)" % (np.mean(np.array(best_f1)), np.std(np.array(best_f1))))

    util.cleanup_cache()
    data_train.purge_cache()
    data_valid.purge_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MIL Cross Val', conflict_handler="resolve")
    parser.add_argument('-c', '--config', default='configs/config_MIL_africa_lymphoma.json', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='0', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-s', '--seed', default='123', type=int,
                        help='seed number (default: 123)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optim;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    start = time.time()
    main(parser, options)
    total_train_time = time.time() - start

    print(f"entire training takes {datetime.timedelta(seconds=int(total_train_time))}")
