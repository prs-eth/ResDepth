from argparse import ArgumentParser
import itertools
import logging
import logging.config
import numpy as np
import os
import random
import sys
import torch

from lib.arguments import INPUT_CHANNELS
from lib.config import cfg as cfg_default
from lib.formatter import RawFormatter
from lib import cfg_utils, fdutil, io_control_file, validate_cfg_training
from lib import utils

PIN_MEMORY = True


def set_seed(seed):
    # Set the random seeds for repeatability
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


parser = ArgumentParser(description='ResDepth:\nA Deep Prior For 3D Reconstruction From High-resolution Satellite '
                                    'Images (Training)',
                        formatter_class=RawFormatter)

parser.add_argument('config_file', type=str, help='JSON configuration file')


def main():
    # Parse the command line arguments
    args = parser.parse_args()
    cfg_file = args.config_file

    prog_name = 'Running ResDepth: Training'
    print('\n{}\n{}\n'.format(prog_name, '=' * len(prog_name)))

    if not fdutil.file_exists(cfg_file):
        print(f'ERROR: Cannot find the configuration file: {cfg_file}')
        sys.exit(1)

    # Read the user configuration file
    cfg_user = cfg_utils.read_json(cfg_file)

    if not cfg_user:
        sys.exit(1)

    # Create the output directory. The name of the output directory is a combination of the current date, time, and the
    # suffix specified in the configuration dictionary.
    output_directory = utils.create_output_directory(cfg_user)

    # Set up Logger
    log_file = os.path.join(output_directory, 'run.log') if output_directory else None
    logger = utils.setup_logger('root_logger', level=logging.INFO, log_to_console=True, log_file=log_file)

    # Verify the user configuration file
    logger.info(f'Validate the configuration file:\t{cfg_file}\n\n')
    if validate_cfg_training.validate_cfg_file(cfg_user, logger) is False:
        sys.exit(1)

    # Complement missing optional keys for each dataset. The value of a key is derived from
    # a) the user config file, if the respective key is universally defined in the general settings
    #    (i.e., same value for each dataset)
    # b) the default config file, if the respective key is not found in the user config file
    validate_cfg_training.augment_dataset_args(cfg_user)

    # Update the default configuration dictionary (cfg_default) with the user configuration dictionary (cfg_user).
    # The default dictionary is defined in the global variable 'cfg' in lib/config.py.
    cfg = cfg_utils.merge(cfg_default, cfg_user)

    # Remove obsolete keys
    cfg_utils.remove_obsolete_keys(cfg)

    # Save output directories
    cfg.output.output_directory = output_directory
    cfg.output.tboard_log_dir = os.path.join(cfg.output.tboard_log_dir, os.path.basename(output_directory))

    # Print input arguments to console
    logger.info('\n\nSettings\n--------\n')
    cfg_utils.print_json(cfg, logger=logger)

    if cfg.general.random_seed is not None:
        set_seed(cfg.general.random_seed)

    # -------------------------- Data allocation, normalization parameters -------------------------- #
    # Input channel configurations with at least one input image (in addition to depth/height map, i.e. the initial DSM)
    channels = INPUT_CHANNELS.copy()
    channels.remove('geom')

    logger.info('\n\n\nData initialization\n-------------------\n')
    if cfg.model.input_channels != 'geom':
        logger.info('Read image pairs...\n')
        if utils.read_image_pairs(cfg, logger) is False:
            sys.exit(1)

    logger.info('Perform data allocation...\n')
    utils.allocate_area(cfg)

    # Extract the definition of the training and validation datasets
    cfg_traindata = utils.prepare_dataset_configuration(cfg, phase='train')
    cfg_valdata = utils.prepare_dataset_configuration(cfg, phase='val')

    logger.info('\n\nData normalization\n-------------------\n')

    # Compute normalization parameters of the DSM(s)
    logger.info('Compute standard deviation over all centered DSM training patches...\n')
    dataloader = utils.get_dataloader(cfg_traindata, sampling_strategy='train',
                                      transform_dsm=False, transform_orthos=False,
                                      use_all_stereo_pairs=False, permute_images_within_pair=False,
                                      input_channels=cfg.model.input_channels,
                                      tile_size=cfg.training_settings.tile_size, augment=False, batch_size=1,
                                      shuffle=False, workers=cfg.general.workers, pin_memory=PIN_MEMORY)

    dsm_std = utils.compute_local_dsm_std_per_centered_patch(dataloader)
    logger.info('Standard deviation:\t{:.3f} m\n'.format(dsm_std))

    for dataset in itertools.chain(cfg_traindata, cfg_valdata):
        dataset.dsm_mean = None
        dataset.dsm_std = dsm_std

    # Compute normalization parameters of the ortho-images
    if cfg.model.input_channels in channels:
        logger.info('\nCompute satellite image normalization parameters...\n')
        images_mean, images_std = utils.compute_satellite_image_normalization(cfg_traindata)
        logger.info('Mean:\t\t\t{:.3f}'.format(images_mean))
        logger.info('Standard deviation:\t{:.3f}\n'.format(images_std))

        for dataset in itertools.chain(cfg_traindata, cfg_valdata):
            dataset.images_mean = images_mean
            dataset.images_std = images_std
    else:
        for dataset in itertools.chain(cfg_traindata, cfg_valdata):
            dataset.images_mean = None
            dataset.images_std = None

    # ----------------------------------------- Dataloaders ----------------------------------------- #
    logger.info('\nInitialize data loaders...\n')

    if cfg.general.random_seed is not None:
        set_seed(cfg.general.random_seed)

    trainloader = utils.get_dataloader(cfg_traindata, sampling_strategy='train',
                                       transform_dsm=True, transform_orthos=True,
                                       use_all_stereo_pairs=cfg.stereopair_settings.use_all_stereo_pairs,
                                       permute_images_within_pair=cfg.stereopair_settings.permute_images_within_pair,
                                       input_channels=cfg.model.input_channels,
                                       tile_size=cfg.training_settings.tile_size, augment=cfg.training_settings.augment,
                                       batch_size=cfg.training_settings.batch_size, shuffle=True,
                                       workers=cfg.general.workers, pin_memory=PIN_MEMORY)

    valloader = utils.get_dataloader(cfg_valdata, sampling_strategy='val',
                                     transform_dsm=True, transform_orthos=True,
                                     use_all_stereo_pairs=True, permute_images_within_pair=False,
                                     input_channels=cfg.model.input_channels,
                                     tile_size=cfg.training_settings.tile_size, augment=False,
                                     batch_size=cfg.training_settings.batch_size, shuffle=False,
                                     workers=cfg.general.workers, pin_memory=PIN_MEMORY)

    # ----------------------- Prepare output folders and write control files ----------------------- #
    logger.info('\nPrepare output folders and files\n--------------------------------\n')

    # Create a subdirectory within the result directory (the name of the subdirectory consists of the code execution
    # day and time and an optional user-specified suffix)
    fdutil.make_dir(cfg.output.output_directory)

    # Save the path of the checkpoint directory
    cfg.output.checkpoint_dir = os.path.join(cfg.output.output_directory, 'checkpoints')
    logger.info(f"\nModel weights will be stored in:\n{cfg.output.checkpoint_dir}\n")

    # Create a pickle file to store the DSM normalization parameters
    cfg.output.dsm_normalization = os.path.join(cfg.output.output_directory, 'DSM_normalization_parameters.p')
    logger.info(f"Writing DSM normalization parameters to file:\n{cfg.output.dsm_normalization}\n")
    io_control_file.write_normalization_params_to_file(cfg.output.dsm_normalization, None, dsm_std)

    # Create a pickle file to store the satellite image normalization parameters
    if cfg.model.input_channels in channels:
        cfg.output.satellite_image_normalization = os.path.join(cfg.output.output_directory,
                                                                'Image_normalization_parameters.p')
        logger.info(f"Writing satellite image normalization parameters to file:"
                    f"\n{cfg.output.satellite_image_normalization}\n")
        io_control_file.write_normalization_params_to_file(cfg.output.satellite_image_normalization, images_mean,
                                                           images_std)

    # Write the final configuration to file
    config_file = os.path.join(cfg.output.output_directory, 'config.json')
    cfg_utils.write_json(cfg, config_file)

    # Write the original user input configuration to file
    config_file = os.path.join(cfg.output.output_directory, 'config.json.orig')
    cfg_utils.write_json(cfg_user, config_file)
    del cfg_user

    # --------------------------------------- Define the model --------------------------------------- #
    logger.info('\nPrepare training\n----------------\n')
    model, args_model = utils.get_model(cfg, logger)

    # Log model parameters to file
    config_file = os.path.join(cfg.output.output_directory, 'model_config.json')
    cfg_utils.write_json(args_model, config_file)

    # Write model architecture to txt file
    if cfg.output.plot_model_txt:
        file = os.path.join(cfg.output.output_directory, 'model_parameters.txt')
        logger.info(f'Writing model architecture to file: {file}\n')
        utils.write_model_structure_to_file(file, model, cfg.training_settings.tile_size,
                                            args_model.settings.n_input_channels)

    # Get the optimizer, the loss function, and the learning rate scheduler
    optimizer = utils.get_optimizer(cfg, model, logger)
    criterion = utils.get_loss(cfg, logger)
    scheduler = utils.get_scheduler(cfg, optimizer, logger)

    # ------------------------------------------- Training ------------------------------------------- #
    # Get trainer and start training
    trainer = utils.get_trainer(cfg, trainloader, valloader, model, optimizer, scheduler, criterion)
    trainer.train()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
    else:
        main()
