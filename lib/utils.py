from datetime import datetime
from easydict import EasyDict as edict
import glob
import itertools
import logging
import numpy as np
import os
from pathlib import Path
import shutil
import re
import sys
import torch
import torchsummary

from lib import arguments, data_allocation, fdutil, io_control_file, rasterutils
from lib.config import cfg as cfg_default
from lib.DsmOrthoDataset import DsmOrthoDataset
from lib.formatter import LeveledFormatter
from lib.Trainer import Trainer
from lib.UNet import UNet


def create_output_directory(cfg):
    """
    Creates the output directory.

    :param cfg: EasyDict, json configuration file imported as dictionary
    :return:    str, path of the output directory
    """

    if 'output' in cfg and 'output_directory' in cfg.output and isinstance(cfg.output.output_directory, str):
        if 'suffix' in cfg.output and isinstance(cfg.output.suffix, str):
            # Use user-defined suffix
            name = create_output_folder_name(cfg.output.suffix)
        else:
            # Use default suffix
            name = create_output_folder_name(cfg_default.output.suffix)

        output_directory = os.path.join(cfg.output.output_directory, name)
        fdutil.make_dir(output_directory)
    else:
        output_directory = None

    return output_directory


def create_output_folder_name(suffix=None):
    """
    Creates the name of the output folder. The name is a combination of the current date, time, and an optional suffix.

    :param suffix: str, folder name suffix
    :return:       str, name of the output directory
    """

    # Record start execution date and time
    now = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # Prepare path to subdirectory within the result directory
    name = '_'.join([now, suffix]) if suffix else now

    return name


def print_dataset_name_to_console(dataset, index, logger=None):
    """
    Prints the dataset name to console.

    :param dataset: dict, configuration of a single dataset
    :param index:   int, index of the dataset
    :param logger:  logger instance (if None, output is print to console)
    """

    name = 'Dataset ' + str(index) + ': ' + str(dataset.name) if 'name' in dataset else 'Dataset ' + str(index) + ':'

    if logger:
        logger.info('{}\n{}\n'.format(name, '~' * len(name)))
    else:
        print('{}\n{}\n'.format(name, '~' * len(name)))


def allocate_area(cfg):
    """
    Splits each dataset into geographically separate stripes for training, validation, and testing. The geographic
    definition of each training, validation, and test region is stored in-place in cfg.

    :param cfg:     EasyDict, json configuration file imported as dictionary
    """

    # Iterate over each dataset configuration
    for i, dataset in enumerate(cfg.datasets):
        if dataset.allocation_strategy == 'entire':
            # Use the entire raster
            extent = rasterutils.get_raster_extent(cfg.datasets[i].raster_in)
            dataset.area_defn = edict({'x_extent': [(0, extent['cols'] - 1)], 'y_extent': [(0, extent['rows'] - 1)]})
        else:
            crossval_training = dataset.crossval_training if 'crossval_training' in dataset else False

            train_area, val_area, test_area = data_allocation.allocate_data(dataset.raster_in,
                                                                            dataset.allocation_strategy,
                                                                            test_stripe=dataset.test_stripe,
                                                                            crossval_training=crossval_training)

            if 'train' in dataset.area_type:
                dataset.area_defn_train = edict(train_area)
            if 'val' in dataset.area_type:
                dataset.area_defn_val = edict(val_area)
            if 'test' in dataset.area_type:
                dataset.area_defn = edict(test_area)


def compute_local_dsm_std_per_centered_patch(dataloader, raster_identifier='raster_in'):
    """
    Computes a single, robust scale factor across all DSM training data samples, so as to preserve a (relative) notion
    of scale. The function first centers each training DSM patch to its mean height. Then, it computes the standard
    deviation of the height within each patch, discards standard deviations below the 5th percentile and above the 95th
    percentile to ensure robustness, and averages the remaining ones to obtain a single, robust estimate of the
    standard deviation.

    :param dataloader:          torch.utils.data.DataLoader instance
    :param raster_identifier:   str, identifier to select the DSM data source
                                (default: 'raster_in' to use the initial DSM)
    :return:                    float, standard deviation of the zero-centered DSM training patches
    """

    # Extract the number of batches
    n_batches = dataloader.__len__()

    # Initialize buffers
    stds = np.zeros(n_batches, dtype=float)

    # Compute the standard deviation over all training pixels per batch (= per single sample)
    for i, batch in enumerate(dataloader):
        if raster_identifier == 'raster_in':
            x = batch['input'][:, 0, :, :].numpy().astype(np.float128)
        else:
            x = batch['target'][:, 0, :, :].numpy().astype(np.float128)

        # Extract the nodata value
        nodata = batch['nodata'].numpy()

        if len(set(nodata)) == 1:   # same nodata value for each sample in the batch
            x = np.ma.masked_where(x == nodata[0], x)
        else:
            x = np.ma.masked_array(x)

            for j in range(x.shape[0]):
                x[j, ...] = np.ma.masked_where(x[j, ...] == nodata[j], x[j, ...])

        mean_per_sample = x.mean(axis=(1, 2), keepdims=True)
        stds[i] = np.sqrt(((x - mean_per_sample) ** 2).sum() / (x.count() - 1))

    # Discard standard deviations below the 5th percentile and above the 95th percentile to ensure robustness and
    # average the remaining ones to obtain a single, robust estimate of the standard deviation
    perc95 = np.percentile(stds, 95)
    perc5 = np.percentile(stds, 5)
    std = stds[np.logical_and(stds >= perc5, stds <= perc95)].mean().item()

    return std


def compute_satellite_image_normalization(cfg_data):
    """
    Computes the mean radiance and its standard deviation across all ortho-images used during training.

    :param cfg_data:    list of EasyDict dictionaries, collection of dictionaries to setup a torch.utils.data.DataLoader
                        instance; output of the function prepare_dataset_configuration()
    :return:            two floats, mean radiance and its standard deviation across all ortho-images used during training
    """

    list_data = []

    for i, dataset in enumerate(cfg_data):
        # Extract the indices of the training images from the list of image pairs and avoid duplicates
        image_ids = list(set(itertools.chain(*dataset.image_pairs)))

        for index in image_ids:
            # Load image
            ds = rasterutils.load_raster(dataset.image_list[index])
            img = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            ds = None

            # Extract the area definition
            area_defn = dataset.area_defn

            for j in range(len(area_defn.x_extent)):
                # Extent of the j.th training region of the i.th dataset
                x = area_defn.x_extent[j]
                y = area_defn.y_extent[j]

                # Extract the image patch corresponding to the training region
                list_data.append(img[y[0]:y[1] + 1, x[0]:x[1] + 1].flatten())

    # Flatten the list of image numpy arrays
    data = np.concatenate(list_data, axis=0)

    # Compute normalization
    mean = np.mean(data).item()
    std = np.std(data).item()

    return mean, std


def get_dataloader(cfg_data, sampling_strategy, transform_dsm, transform_orthos, use_all_stereo_pairs,
                   permute_images_within_pair, input_channels, tile_size, augment, batch_size, shuffle, workers,
                   pin_memory):
    """
    Returns a data loader using the dataset configurations specified in cfg_data.

    :param cfg_data:                    list of EasyDict dictionaries, collection of dictionaries to setup a
                                        torch.utils.data.DataLoader instance; output of the function
                                        prepare_dataset_configuration()
    :param sampling_strategy:           str, choose among ['train', 'val', 'test']
    :param transform_dsm:               bool, True to normalize the DSMs, False otherwise
    :param transform_orthos:            bool, True to normalize the ortho-images, False otherwise
    :param use_all_stereo_pairs:        bool, True to combine each image pair in cfg_data[i].image_pairs
                                        (of the i.th dataset) with the same initial DSM patch to form a data sample;
                                        False to randomly assign an image pair in cfg_data[i].image_pairs to each
                                        initial DSM patch
    :param permute_images_within_pair:  bool, True to randomly flip the order of the ortho-images in every training
                                        sample; False to maintain the order of the ortho-images according to the
                                        definition in cfg_data[i].image_pairs
    :param input_channels:              str, input channel configuration (model architecture),
                                        choose among ['geom-multiview', 'geom-stereo', 'geom-mono', 'stereo', 'geom']
    :param tile_size:                   int, tile size in pixels
    :param augment:                     bool, True to activate data augmentation (random rotation by multiples of
                                        90 degrees as well as random flipping along the horizontal and vertical axes),
                                        False otherwise
    :param batch_size:                  int, batch size
    :param shuffle:                     bool, True to reshuffle the data samples at every epoch, False otherwise
    :param workers:                     int, number of workers
    :param pin_memory:                  bool, if True, the data loader will copy Tensors into CUDA pinned memory before
                                        returning them
    :return:                            torch.utils.data.DataLoader instance
    """

    assert sampling_strategy in ['train', 'val', 'test']

    list_dsets = []

    # Iterate over each dataset configuration
    for dataset in cfg_data:
        if transform_dsm is True:
            dsm_mean = dataset.dsm_mean
            dsm_std = dataset.dsm_std
        else:
            dsm_mean = None
            dsm_std = 1.0

        if transform_orthos is True and input_channels != 'geom':
            images_mean = dataset.images_mean
            images_std = dataset.images_std
        else:
            images_mean = None
            images_std = 1.0

        dset = DsmOrthoDataset(dataset, input_channels=input_channels, tile_size=tile_size,
                               sampling_strategy=sampling_strategy,
                               transform_dsm=transform_dsm, transform_orthos=transform_orthos,
                               dsm_mean=dsm_mean, dsm_std=dsm_std, ortho_mean=images_mean, ortho_std=images_std,
                               augment=augment, use_all_stereo_pairs=use_all_stereo_pairs,
                               permute_images_within_pair=permute_images_within_pair)
        list_dsets.append(dset)

    if len(list_dsets) > 1:
        dsets = torch.utils.data.ConcatDataset(list_dsets)
    else:
        dsets = dset

    dataloader = torch.utils.data.DataLoader(dsets, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                             pin_memory=pin_memory)

    return dataloader


def get_loss(cfg, logger=None):
    """
    Returns the loss function.

    :param cfg:         EasyDict, json configuration file imported as dictionary
    :param logger:      logger instance (if None, output is print to console)
    :return:            torch.nn instance, loss function used for training
    """

    if cfg.training_settings.loss == 'L1':
        criterion = torch.nn.L1Loss(reduction='mean')
    else:
        if logger:
            logger.error(f"{cfg.training_settings.loss} loss is not implemented. Choose among {arguments.LOSSES}.\n")
        else:
            print(f"ERROR: {cfg.training_settings.loss} loss is not implemented. Choose among {arguments.LOSSES}.\n")

    return criterion


def get_model(cfg, logger=None):
    """
    Returns a model instance.

    :param cfg:         EasyDict, json configuration file imported as dictionary
    :param logger:      logger instance (if None, output is print to console)

    :return model:      nn.Module, the model used for training
    :return args_model: EasyDict, dictionary storing the model architecture parameters
    """

    args_model = _collect_model_args(cfg)

    if args_model.name == 'UNet':
        model = UNet(**args_model.settings)
    else:
        if logger:
            logger.error(f"{args_model.name} model is not implemented. Choose among {arguments.ARCHITECTURES}.\n")
        else:
            print(f"ERROR: {args_model.name} model is not implemented. Choose among {arguments.ARCHITECTURES}.\n")

    return model, args_model


def get_optimizer(cfg, model, logger=None):
    """
    Returns an optimizer instance.

    :param cfg:         EasyDict, json configuration file imported as dictionary
    :param model:       nn.Module instance, the model used for training
    :param logger:      logger instance (if None, output is print to console)
    :return:            torch.optim.optimizer instance, optimizer used for training
    """

    if cfg.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.learning_rate,
                                     weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.learning_rate,
                                    weight_decay=cfg.optimizer.weight_decay)
    else:
        if logger:
            logger.error(f"{cfg.optimizer.name} optimizer is not implemented. Choose among {arguments.OPTIMIZERS}.\n")
        else:
            print(f"ERROR: {cfg.optimizer.name} optimizer is not implemented. Choose among {arguments.OPTIMIZERS}.\n")

    return optimizer


def get_scheduler(cfg, optimizer, logger=None):
    """
    Returns a learning rate scheduler instance.

    :param cfg:       EasyDict, json configuration file imported as dictionary
    :param optimizer: torch.optim.optimizer instance, optimizer used to train the network
    :param logger:    logger instance (if None, output is print to console)
    :return:          torch.optim.lr_scheduler instance, learning rate scheduler
                      (None, if the learning rate scheduler is disabled)
    """

    if cfg.scheduler.enabled:
        name = cfg.scheduler.name

        if name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True,
                                                                   **cfg.scheduler.settings)
        elif name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, verbose=False, **cfg.scheduler.settings)

        elif name == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, verbose=False, **cfg.scheduler.settings)

        else:
            if logger:
                logger.error(f"{name} learning rate scheduler is not implemented."
                             f"Choose among {arguments.SCHEDULERS}.\n")
            else:
                print(f"ERROR: {name} learning rate scheduler is not implemented. "
                      f"Choose among {arguments.SCHEDULERS}.\n")
    else:
        scheduler = None

    return scheduler


def get_trainer(cfg, trainloader, valloader, model, optimizer, scheduler, criterion):
    """
    Returns a Trainer instance.

    :param cfg:             EasyDict, json configuration file imported as dictionary
    :param trainloader:     torch.utils.data.DataLoader instance, training data
    :param valloader:       torch.utils.data.DataLoader instance, validation data
    :param model:           nn.Module instance, the model used for training
    :param optimizer:       torch.optim.optimizer instance, optimizer used for training
    :param scheduler:       torch.optim.lr_scheduler instance, learning rate scheduler
                            (None, if the learning rate scheduler is disabled)
    :param criterion:       torch.nn instance, loss function used for training
    :return:                instance of the Trainer class
    """

    config = edict()
    config.trainloader = trainloader
    config.valloader = valloader
    config.model = model
    config.optimizer = optimizer
    config.scheduler = scheduler
    config.criterion = criterion

    config.n_epochs = cfg.training_settings.n_epochs
    config.evaluate_rate = cfg.general.evaluate_rate
    config.save_model_rate = cfg.general.save_model_rate
    config.freq_average_train_loss = 20

    config.save_dir = cfg.output.output_directory
    config.log_file = os.path.join(config.save_dir, 'training.log')
    config.checkpoint_dir = cfg.output.checkpoint_dir
    config.tboard_log_dir = cfg.output.tboard_log_dir

    fdutil.make_dir(config.tboard_log_dir)

    if 'pretrained_path' in cfg.model:
        config.pretrained_path = cfg.model.pretrained_path

        # Get the tensorboard logs directory of the pretrained model
        experiment_directory = Path(config.pretrained_path).parent.parent
        experiment_tboard_log_dir = experiment_directory.parent / 'logs' / experiment_directory.name

        # Find the previous tensorboard files and copy them to the new experiments output folder
        if os.path.isdir(experiment_tboard_log_dir):
            tb_files = glob.glob(os.path.join(experiment_tboard_log_dir, 'events.*'))
            for tb_file in tb_files:
                shutil.copy(tb_file, Path(config.tboard_log_dir) / Path(tb_file).name)

        # Find the previous training log file and copy it to the new experiments output folder
        log_file = experiment_directory / 'training.log'
        if fdutil.file_exists(log_file):
            shutil.copy(log_file, config.log_file)

        # Copy the best model weights so far
        path_model = Path(config.pretrained_path).parents[0] / 'Model_best.pth'
        if fdutil.file_exists(path_model):
            shutil.copy(path_model, Path(config.checkpoint_dir) / 'Model_best.pth')

    else:
        config.pretrained_path = None

    return Trainer(config)


def prepare_dataset_configuration(cfg, phase):
    """
    For each dataset in cfg.datasets that matches the specified phase (dataset to be used for training, validation,
    or testing), this function extracts the relevant key-value pairs that will later be used to define a
    a torch.utils.data.DataLoader instance (per dataset or per phase).

    :param cfg:     EasyDict, json configuration file imported as dictionary
    :param phase:   str, choose among ['train', 'val', 'test'] to extract the definition of the training, validation,
                    or test datasets
    :return:        list of EasyDict dictionaries, each dictionary specifies the relevant key-value pairs per phase
                    dataset to define a data loader instance
    """

    assert phase in ['train', 'val', 'test']

    if phase == 'test':
        if cfg.model.input_channels == 'geom':
            keys = ['name', 'raster_gt', 'raster_in', 'mask_ground_truth', 'mask_building', 'mask_water', 'mask_forest',
                    'area_defn']
        else:
            keys = ['name', 'raster_gt', 'raster_in', 'mask_ground_truth', 'mask_building', 'mask_water', 'mask_forest',
                    'area_defn', 'image_list', 'image_pairs']

        cfg_list = []

        # Iterate over each dataset configuration
        for i, dataset in enumerate(cfg.datasets):
            cfg_dataloader = edict()

            # Extract the relevant key-value pairs required to define a data loader (test data)
            for key in keys:
                if key in dataset:
                    cfg_dataloader[key] = dataset[key]

            if 'mask_ground_truth' not in cfg_dataloader:
                cfg_dataloader.mask_ground_truth = None
            if 'mask_building' not in cfg_dataloader:
                cfg_dataloader.mask_building = None
            if 'mask_water' not in cfg_dataloader:
                cfg_dataloader.mask_water = None
            if 'mask_forest' not in cfg_dataloader:
                cfg_dataloader.mask_forest = None

            # Required if test.py is run on training/validation area (area_type == 'train' or area_type == 'val')
            if 'area_defn_train' in dataset:
                cfg_dataloader.area_defn = dataset['area_defn_train']
            if 'area_defn_val' in dataset:
                cfg_dataloader.area_defn = dataset['area_defn_val']

            cfg_list.append(cfg_dataloader)

    else:
        if cfg.model.input_channels == 'geom':
            keys = ['name', 'raster_gt', 'raster_in']
        else:
            keys = ['name', 'raster_gt', 'raster_in', 'image_list']

        cfg_list = []

        # Iterate over each dataset configuration
        for i, dataset in enumerate(cfg.datasets):
            cfg_dataloader = edict()

            # Extract the relevant key-value pairs required to define a data loader (training or validation data)
            if phase in dataset.area_type:
                for key in keys:
                    if key in dataset:
                        cfg_dataloader[key] = dataset[key]

                if phase == 'train':
                    if cfg.model.input_channels != 'geom':
                        cfg_dataloader.image_pairs = dataset.image_pairs_train
                    cfg_dataloader.area_defn = dataset.area_defn_train
                    cfg_dataloader.n_samples = dataset.n_training_samples

                elif phase == 'val':
                    if cfg.model.input_channels != 'geom':
                        cfg_dataloader.image_pairs = dataset.image_pairs_val
                    cfg_dataloader.area_defn = dataset.area_defn_val

                cfg_list.append(cfg_dataloader)

    return cfg_list


def read_image_pairs(cfg, logger=None):
    """
    Reads and validates the input image (pairs) for ResDepth. The image list and image pairs are stored in-place
    in cfg.

    :param cfg:     EasyDict, json configuration file imported as dictionary
    :param logger:  logger instance (if None, output is print to console)

    :return:        bool, True if no errors have been detected while reading the image pairs, False otherwise
    """

    if logger is None:
        logger = setup_logger('read_image_pairs', log_to_console=True, log_file=None)

    if cfg.model.input_channels != 'geom':
        # Iterate over each dataset configuration
        for i, dataset in enumerate(cfg.datasets):
            print_dataset_name_to_console(dataset, i, logger)

            # Read the image pairs for training
            if 'path_pairlist_training' in dataset:
                dataset.image_list, dataset.image_pairs_train = io_control_file.read_pairlist_from_file(
                    dataset.path_image_list, dataset.path_pairlist_training, logger)

                if dataset.image_pairs_train is None:
                    return False

                if cfg.model.input_channels == 'geom-multiview':
                    multiview_config = cfg.multiview.config
                else:
                    multiview_config = None

                if not _valid_image_pairs(cfg.model.input_channels, dataset.path_pairlist_training,
                                          dataset.image_pairs_train, multiview_config, logger):
                    return False

                if len(dataset.image_pairs_train) > 1:
                    logger.info('Selected the following image pairs for training:')
                else:
                    if len(dataset.image_pairs_train[0]) > 1:
                        logger.info('Selected the following image pair for training:')
                    else:
                        logger.info('Selected the following image for training:')
                for pair in dataset.image_pairs_train:
                    names = [fdutil.filename(dataset.image_list[x]) for x in pair]
                    logger.info(', '.join(names))
                logger.info('\n')

            # Read the image pairs for validation
            if 'path_pairlist_validation' in dataset:
                _, dataset.image_pairs_val = io_control_file.read_pairlist_from_file(dataset.path_image_list,
                                                                                     dataset.path_pairlist_validation,
                                                                                     logger)

                if dataset.image_pairs_val is None:
                    return False

                if cfg.model.input_channels == 'geom-multiview':
                    multiview_config = cfg.multiview.config
                else:
                    multiview_config = None

                if not _valid_image_pairs(cfg.model.input_channels, dataset.path_pairlist_validation,
                                          dataset.image_pairs_val, multiview_config, logger):
                    return False

                if len(dataset.image_pairs_val) > 1:
                    logger.info('Selected the following image pairs for validation:')
                else:
                    if len(dataset.image_pairs_val[0]) > 1:
                        logger.info('Selected the following image pair for validation:')
                    else:
                        logger.info('Selected the following image for validation:')
                for pair in dataset.image_pairs_val:
                    names = [fdutil.filename(dataset.image_list[x]) for x in pair]
                    logger.info(', '.join(names))
                logger.info('\n')

            # Read the image pairs for testing
            if 'path_pairlist' in dataset:
                dataset.image_list, dataset.image_pairs = io_control_file.read_pairlist_from_file(
                    dataset.path_image_list, dataset.path_pairlist, logger)

                if dataset.image_pairs is None:
                    return False

                if cfg.model.input_channels == 'geom-multiview':
                    n_views = cfg.model.settings.n_input_channels - 1
                    multiview_config = f'{n_views}-view'
                else:
                    multiview_config = None

                if not _valid_image_pairs(cfg.model.input_channels, dataset.path_pairlist, dataset.image_pairs,
                                          multiview_config, logger):
                    return False

                if len(dataset.image_pairs) > 1:
                    logger.info('The following image pairs will be used for inference:')
                else:
                    if len(dataset.image_pairs[0]) > 1:
                        logger.info('The following image pair will be used for inference:')
                    else:
                        logger.info('The following image will be used for inference:')
                for pair in dataset.image_pairs:
                    names = [fdutil.filename(dataset.image_list[x]) for x in pair]
                    logger.info(', '.join(names))
                logger.info('\n')

    return True


def setup_logger(logger_name, level=logging.INFO, log_to_console=True, log_file=None):
    """
    Returns a logger.

    :param logger_name:     str, name of the logger
    :param level:           sets the logger level to the specified level
    :param log_to_console:  bool, True to add a StreamHandler
    :param log_file:        str, filename of the FileHandler
    :return:                logger instance
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = LeveledFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    formatter.set_formatter(logging.INFO, logging.Formatter('%(message)s'))
    formatter.set_formatter(logging.WARNING, logging.Formatter('%(levelname)s: %(message)s'))
    formatter.set_formatter(logging.ERROR, logging.Formatter('%(levelname)s: %(message)s'))

    if log_to_console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def add_console_logger(logger):
    """
    Adds a StreamHandler to the logger instance.

    :param logger:  logger instance
    """

    formatter = LeveledFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    formatter.set_formatter(logging.INFO, logging.Formatter('%(message)s'))
    formatter.set_formatter(logging.WARNING, logging.Formatter('%(levelname)s: %(message)s'))
    formatter.set_formatter(logging.ERROR, logging.Formatter('%(levelname)s: %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def add_file_logger(logger, log_file):
    """
    Adds a FileHandler to the logger instance.

    :param logger:  logger instance
    :param log_file:        str, filename of the FileHandler
    """

    formatter = LeveledFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    formatter.set_formatter(logging.INFO, logging.Formatter('%(message)s'))
    formatter.set_formatter(logging.WARNING, logging.Formatter('%(levelname)s: %(message)s'))
    formatter.set_formatter(logging.ERROR, logging.Formatter('%(levelname)s: %(message)s'))

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)



def write_model_structure_to_file(filepath, model, image_size, in_channels):
    """
    Writes the model architecture to a text file.

    :param filepath:        str, path to the output text file
    :param model:           nn.Module instance, the model used for training
    :param image_size:      int, tile size in pixels
    :param in_channels:     int, number of input channels
    """

    # Redirect stdout to file
    original = sys.stdout
    sys.stdout = open(filepath, "w")
    torchsummary.summary(model, input_size=(in_channels, image_size, image_size), device="cpu")
    print('\n\n')
    print(model)

    # Reset stdout
    sys.stdout = original


def _collect_model_args(cfg):
    """
    Extracts the model architecture parameters from the general settings dictionary cfg.

    :param cfg: EasyDict, json configuration file imported as dictionary
    :return:    EasyDict, dictionary storing the model parameters
    """

    args_model = edict({'name': cfg.model.name, 'input_channels': cfg.model.input_channels, 'settings': edict()})

    if cfg.model.name == 'UNet':
        args_model.settings.n_input_channels = _count_number_of_input_channels(cfg)
        args_model.settings.start_kernel = cfg.model.start_kernel
        args_model.settings.depth = cfg.model.depth
        args_model.settings.act_fn_encoder = cfg.model.act_fn_encoder
        args_model.settings.act_fn_decoder = cfg.model.act_fn_decoder
        args_model.settings.act_fn_bottleneck = cfg.model.act_fn_bottleneck
        args_model.settings.up_mode = cfg.model.up_mode
        args_model.settings.do_BN = cfg.model.do_BN
        args_model.settings.outer_skip = cfg.model.outer_skip
        args_model.settings.outer_skip_BN = cfg.model.outer_skip_BN
        args_model.settings.bias_conv_layer = cfg.model.bias_conv_layer

    return args_model


def _count_number_of_input_channels(cfg):
    """
    Returns the number of input channels of ResDepth.

    :param cfg: EasyDict, json configuration file imported as dictionary
    :return:    int, number of input channels
    """

    input_channels = cfg.model.input_channels

    if input_channels == 'geom':
        return 1

    elif input_channels in ['stereo', 'geom-mono']:
        return 2

    elif input_channels == 'geom-stereo':
        return 3

    elif input_channels == 'geom-multiview':
        num_views = int(re.findall(r'\d+', cfg.multiview.config)[0])
        return num_views + 1


def _valid_image_pairs(input_channels, pairlist, image_pairs, multiview_config=None, logger=None):
    """
    Verifies that the number of images forming an image pair is in accordance with the specified input channel
    configuration of the network.

    :param input_channels:      str, input channel configuration (model architecture)
    :param pairlist:            str, path of the text file that specifies the image pair(s)
    :param image_pairs:         list of tuples, image pairs imported from pairlist
    :param multiview_config:    str, multi-view configuration
                                (None if mono/stereo guidance or no image guidance at all)
    :param logger:              logger instance (if None, output is print to console)
    :return:
    """

    error = False

    if logger is None:
        logger = setup_logger('_valid_image_pairs', log_to_console=True, log_file=None)

    if input_channels == 'geom-multiview':
        num_views = int(re.findall(r'\d+', multiview_config)[0])
        if num_views != len(image_pairs[0]):
            logger.error("Argument 'model': 'input_channels' is set to 'geom-multiview' and 'multiview': 'config' "
                         f"to '{multiview_config}'. Provide (an) image pair(s) composed of {num_views} images instead "
                         f"of {len(image_pairs[0])} images in {pairlist}.\n")
            error = True

    elif 'stereo' in input_channels:
        if len(image_pairs[0]) != 2:
            logger.error(f"Argument 'model': 'input_channels' is set to '{input_channels}'. "
                         f"Provide image pairs composed of 2 images in {pairlist}.\n")
            error = True

    elif input_channels == 'geom-mono':
        if len(image_pairs) != 1:
            logger.error(f"Argument 'model': 'input_channels' is set to '{input_channels}'. "
                         f"Provide a single image in {pairlist}.\n")
            error = True

        if len(image_pairs[0]) != 1:
            logger.error(f"Argument 'model': 'input_channels' is set to '{input_channels}'. "
                         f"Provide a single image in {pairlist} instead of image pairs.\n")
            error = True

    return not error
