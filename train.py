import shutup

shutup.please()
import os
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from data.megadepth_datamodule import MegaDepthPairsDataModule
from models.matching_module import MatchingTrainingModule
from utils.train_utils import get_training_loggers, get_training_callbacks, prepare_logging_directory


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--config', type=str, help='path to config file', default='config/config.yaml')
    parser.add_argument('--features_config', type=str, help='path to config file with features', default='config/features_online/sift.yaml')
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load('/content/OpenGlue/config/config.yaml')  # base config
    feature_extractor_config = OmegaConf.load(args.features_config)
    
    if args.config != '/content/OpenGlue/config/config.yaml':
        add_conf = OmegaConf.load(args.config)
        config = OmegaConf.merge(config, add_conf)

    pl.seed_everything(int(os.environ.get('LOCAL_RANK', 0))) # LOCAL_RANK:同じノード内のプロセス中何番目かを表す番号

    # Prepare directory for logs and checkpoints
    if os.environ.get('LOCAL_RANK', 0) == 0:
        experiment_name = '{}__attn_{}__laf_{}__{}'.format(
                                                            feature_extractor_config['name'], # 'SIFT'
                                                            config['superglue']['attention_gnn']['attention'], # softmax
                                                            config['superglue']['laf_to_sideinfo_method'], # 'none'
                                                            str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                                                          )
        
        log_path = prepare_logging_directory(config, experiment_name, features_config=feature_extractor_config)
        
    else:
        experiment_name, log_path = '', ''

    # Init Lightning Data Module
    data_config = config['data']
    dm = MegaDepthPairsDataModule(
                                    root_path=data_config['root_path'],
                                    train_list_path=data_config['train_list_path'],
                                    val_list_path=data_config['val_list_path'],
                                    test_list_path=data_config['test_list_path'],
                                    batch_size=data_config['batch_size_per_gpu'],
                                    num_workers=data_config['dataloader_workers_per_gpu'],
                                    target_size=data_config['target_size'],
                                    val_max_pairs_per_scene=data_config['val_max_pairs_per_scene'],
                                    train_pairs_overlap=data_config.get('train_pairs_overlap')
                                )

    # Init model
    model = MatchingTrainingModule(
                                    train_config={**config['train'], **config['inference'], **config['evaluation']},
                                    features_config=feature_extractor_config,
                                    superglue_config=config['superglue'],
                                    )

    # Set callbacks and loggers
    callbacks = get_training_callbacks(config, log_path, experiment_name)
    loggers = get_training_loggers(config, log_path, experiment_name)

    # Init distributed trainer
    trainer = pl.Trainer(
                        gpus=config['gpus'],
                        max_epochs=config['train']['epochs'],
                        accelerator="gpu", # "ddp", # ハードウェアの種類を指す: "cpu" "gpu" "tpu" "ipu" "auto"
                        gradient_clip_val=config['train']['grad_clip'],
                        log_every_n_steps=config['logging']['train_logs_steps'], # ログをとる頻度。
                        limit_train_batches=config['train']['steps_per_epoch'], # 学習で使用するデータの割合を指定する。デバッグ等で使用する。
                        num_sanity_val_steps=0,
                        callbacks=callbacks,
                        logger=loggers,
                        plugins=DDPPlugin(find_unused_parameters=False),
                        precision=config['train'].get('precision', 32), # 学習の小数の精度を指定。CPUでは16-bitはサポートされていない。
                        )
    
    # If loaded from checkpoint - validate
    if config.get('checkpoint') is not None:
        trainer.validate(model, datamodule=dm, ckpt_path=config.get('checkpoint'))
        
    # datamodule=dm:dm.init() > dm.setup()の順で実行される
    trainer.fit(model, datamodule=dm, ckpt_path=config.get('checkpoint'))


if __name__ == '__main__':
    main()
