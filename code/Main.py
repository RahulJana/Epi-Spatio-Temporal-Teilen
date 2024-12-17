# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings

import torch

import Train
import Datatreat
from datetime import datetime

import Toolkits
from Constant import Paths, DATE
from Logconfig import LoggerManager

logger = LoggerManager.get_logger()

warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")


if __name__ == '__main__':
    """
        # spatial temporal
    """
    start_time = datetime.now()
    logger.info('begin to execute...')
    
    parser = argparse.ArgumentParser(description='Run')

    # command line arguments
    parser.add_argument('-dev', '--dev', type=str, help='Specify GPU usage', default="cuda:0" if torch.cuda.is_available() 
                        else ("cpu" if torch.backends.mps.is_available() else "cpu"))  # replace the 'cpu' with the 'mps' if it is on mac
    
    parser.add_argument('-data_type', '--data_type', type=str, default='germany', help='china, germany, mock')
    parser.add_argument('-in', '--input_dir', type=str, default=Paths.DATA_RAW)
    parser.add_argument('-out', '--output_dir', type=str, default=Paths.DEST_DIR)
    
    parser.add_argument('-split_ratio', '--split_ratio', type=float, nargs='+',
                        help='Data split ratio in [training : validation : test], Example: -split 6 2 2', default=[6, 2, 2])
    parser.add_argument('-window_rolling', '--window_rolling', type=int, default=28)  # The smaller the value, the closer it is.
    parser.add_argument('-model_type', '--model_type', type=str, nargs='+', help='Specify model type', choices=['SSIR_STGCN', 'SSIR_STGAT', 'SSIR_ODEFIT'], default=['SSIR_STGCN'])
    parser.add_argument('-graph_type', '--graph_type', type=str, help='Specify graph type', choices=['Static', 'Dynamic', 'Adaptive'], default='Dynamic')
    
    parser.add_argument('-obs', '--obs_len', type=int, help='Length of observation data', default=7)
    parser.add_argument('-pre', '--pre_len', type=int, help='Length of prediction data', default=7)
    
    parser.add_argument('-batch', '--batch_size', type=int, default=8, help='batch_size, := training_size or validation_size or test_size.')
    parser.add_argument('-kernel_size', '--kernel_size', type=int, default=3)
    parser.add_argument('-num_layers', '--num_layers', type=int, default=3)
        
    parser.add_argument('-loss_type', '--loss_type', type=str, help='Specify loss function', choices=['MSE', 'MAE', 'cMSE', 'cMAE'], default='cMAE')
    parser.add_argument('-optim', '--optimizer', type=str, help='Specify optimizer', choices=['Adam', 'SGD', 'RMSprop'], default='Adam')
    parser.add_argument('-sched', '--scheduler', type=str, help='Specify scheduler', choices=['StepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'], default='StepLR')
    parser.add_argument('-mode', '--mode', type=str, choices=['min', 'max'], default='min')
    parser.add_argument('-patience', '--patience', type=int, default=50)
    parser.add_argument('-factor', '--factor', type=float, default=0.99999)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-gamma', '--gamma', type=float, default=0.9999)
    parser.add_argument('-step_size', '--step_size', type=int, default=50)
    parser.add_argument('-momentum', '--momentum', type=float, default=0.95)
    parser.add_argument('-t_max', '--t_max', type=float, default=50)
    parser.add_argument('-eta_min', '--eta_min', type=float, default=1e-3)
    parser.add_argument('-milestones', '--milestones', type=int, nargs='+', default=[50, 100, 300])
     
    parser.add_argument('-rd', '--random_seed', type=int, default=3)
    parser.add_argument('-clip', '--clip', type=float, default=5.0)
    parser.add_argument('-normalize', '--normalize', type=str, default='norm')  # if normalize, should be normalize_ for file name
    parser.add_argument('-w4pre', '--w4pre', type=int, default=1, help='Weight value for data loss')
    parser.add_argument('-w4phy', '--w4phy', type=int, default=1, help='Weight value for residual loss')
    
    parser.add_argument('-max_epoch', '--max_epoch', type=int, default=1)
    parser.add_argument('-early_stop', '--early_stop', type=int, default=1)
    parser.add_argument('-grad_print', '--grad_print', action='store_true', help='Enable gradient printing') 
    parser.add_argument('-beta_incorporated', '--beta_incorporated', action='store_true', help='Incorporate beta') 
    
    parser.add_argument('-daily', '--daily', action='store_true', help='Enable loss caculate by using daily')
    parser.add_argument('-phyloss4all', '--phyloss4all', action='store_true', help='Enable loss caculate by using all S I R') 
    
    parser.add_argument('-test', '--test', action='store_true', help='Enable testing')
    parser.add_argument('-ssir', '--ssir', type=str, default='sir')

    # dict
    params = parser.parse_args().__dict__ 
    # dynamically calculate window size. 'window_size, :=obs_len + pre_len.'
    params['window_size'] = params['obs_len'] + params['pre_len'] 
    params['max_horizon'] = params['pre_len']
    
    logger.critical(params)
    
    # load the index file
    auxdata_file = os.path.join(Paths.DATA_REPO, '_'.join(filter(None, map(str, [params["data_type"], params["normalize"], params["obs_len"], params["pre_len"], DATE.DATE_SELECTED, 'auxdata.json']))))
    # load the data
    datarepo_file = os.path.join(Paths.DATA_REPO, '_'.join(filter(None, map(str, [params["data_type"], params["normalize"], params["obs_len"], params["pre_len"], DATE.DATE_SELECTED, 'repo.pkl']))))
    
    if os.path.exists(datarepo_file):
        # if the data file exist, load it.
        with open(datarepo_file, 'rb') as f:
            data_repo = torch.load(f, weights_only=False, map_location=torch.device(params["dev"]))

        logger.info(f"data repo loaded from file.")  # training
    else:
        # load data
        data_builder = Datatreat.DataBuilder(params=params)
        data = data_builder.build()

        # process data
        data_processor = Datatreat.DataProcessor(params=params)
        data_repo = data_processor.build_data_repo(data=data)
        
        # save data
        with open(datarepo_file, 'wb') as f:
            torch.save(data_repo, f)
        logger.info("data repo processed and saved to local.")
    
    if os.path.exists(auxdata_file):
        with open(auxdata_file, 'r') as f:
            auxdata = json.load(f)
        logger.info(f"auxiliary data is {auxdata}.")
        DATE.DATE_LIST = auxdata['date_list']
    else:        
        raise FileNotFoundError(f"index file {auxdata_file} does not exist.")
    
    for modeltype in params["model_type"]:
        logger.info(f'{modeltype} training begin...')
        # get model
        trainer = Train.Trainer(params=params, model_type=modeltype)

        # run data_repo: (B,T,N,F)
        # train
        trstart_time = datetime.now()
        if modeltype != "SSIR_ODEFIT":
            trainer.train(data_loader=data_repo, modes=['training', 'validation'])
            logger.info(f'{modeltype} training is finished, time consuming: {Toolkits.elapsed_time(trstart_time)}.')
            
            # test
            if params['test']:
                testart_time = datetime.now()
                logger.info(f'{modeltype} test begin...')
                trainer.test(data_loader=data_repo, modes=['training', 'validation', 'test'], auxdata=auxdata)   
                logger.info(f'{modeltype} test is finished, time consuming: {Toolkits.elapsed_time(testart_time)}.')
        else:
            trainer.ode_estimator(data_loader=data_repo['test'], auxdata=auxdata)
            logger.info(f'{modeltype} estimation is finished, time consuming: {Toolkits.elapsed_time(trstart_time)}.')
            
    logger.info(f'all model {params["model_type"]} execution is finished, time consuming: {Toolkits.elapsed_time(start_time)}.')
    