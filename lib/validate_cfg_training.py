from copy import deepcopy
from easydict import EasyDict as edict
import multiprocessing

from lib import arguments, cfg_utils, fdutil, io_control_file, utils
from lib.config import cfg as cfg_default
from lib.validate_arguments import all_keys_known, is_boolean, is_string, is_positive_integer, valid_act_fn,\
    valid_allocation, valid_tile_size


def validate_cfg_file(cfg_file, logger=None):
    """
    Validates a json configuration file used for training. Specifically, the function
    a) detects unknown keys.
    b) checks that all required keys are defined.
    c) checks that all values are valid.
    d) checks that the input data exists.

    :param cfg_file:  str, Dict or EasyDict; path of the json file or file imported as dictionary returned by
                      the function cfg_utils.read_json()
    :param logger:    logger instance (if None, output is print to console)
    :return:          bool, True if no errors have been detected in the configuration file, False otherwise
    """

    if logger is None:
        logger = utils.setup_logger('validate_cfg_file', log_to_console=True, log_file=None)

    # Load the configuration file
    if isinstance(cfg_file, dict):
        cfg_user = edict(deepcopy(cfg_file))
    elif isinstance(cfg_file, edict):
        cfg_user = deepcopy(cfg_file)
    else:
        cfg_user = cfg_utils.read_json(cfg_file)

    # Verify whether all primary keys are known
    if not all_keys_known(cfg_user, arguments.PRIMARY_KEYS, logger):
        return False

    # Verify that all primary keys are specified
    missing_keys = [k for k in arguments.PRIMARY_KEYS_MANDATORY if k not in cfg_user]
    if len(missing_keys) > 0:
        logger.error('The following keys are missing: {}.\n'.format(','.join(["'{}'".format(x) for x in missing_keys])))
        return False

    # Verify user config file: dataset settings
    process = "Verify 'datasets' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if not _valid_dataset_args(cfg_user, logger):
        return False

    # Verify user config file: model settings
    process = "Verify 'model' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if not _valid_model_args(cfg_user, logger):
        return False

    # Verify user config file: multiview settings
    if 'multiview' in cfg_user:
        process = "Verify 'multiview' arguments"
        logger.info('{}\n{}\n'.format(process, '-' * len(process)))

        if not _valid_multiview_args(cfg_user, logger):
            return False

    # Verify user config file: stereo pair settings
    if 'stereopair_settings' in cfg_user:
        process = "Verify 'stereopair_settings' arguments"
        logger.info('{}\n{}\n'.format(process, '-' * len(process)))

        if not _valid_stereo_args(cfg_user, logger):
            return False

    # Verify user config file: training settings
    process = "Verify 'training_settings' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if not _valid_training_args(cfg_user, logger):
        return False

    # Verify user config file: optimizer settings
    process = "Verify 'optimizer' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if not _valid_optimizer_args(cfg_user, logger):
        return False

    # Verify user config file: learning rate scheduler settings
    process = "Verify 'scheduler' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if not _valid_scheduler_args(cfg_user, logger):
        return False

    # Verify user config file: general settings
    process = "Verify 'general' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if not _valid_general_args(cfg_user, logger):
        return False

    # Verify user config file: output settings
    process = "Verify 'output' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if not _valid_output_args(cfg_user, logger):
        return False

    return True


def augment_dataset_args(cfg_user):
    """
    Complements the missing optional keys for each dataset listed in the user configuration file. The value of a key
    is derived from
        a) the user configuration file, if the respective key is universally defined in the general settings
           (i.e., same value for each dataset)
        b) the default configuration file, if the respective key is not found in the user configuration file
    The user configuration file is modified in-place.

    :param cfg_user:  EasyDict, json user configuration file imported as dictionary
    """

    # Global keys to be assigned to each dataset
    keys = arguments.DATASET_KEYS_OPTIONAL.copy()
    unwanted = ['name', 'path_image_list', 'path_pairlist_training', 'path_pairlist_validation', 'crossval_training']
    for elem in unwanted:
        keys.remove(elem)

    # Check whether the above keys were globally defined in the the user configuration file (i.e., the same value for
    # each dataset) If not, extract the default values from the default configuration file.
    settings = edict()
    for key in keys:
        if key == 'n_training_samples':
            if key in cfg_user.training_settings:
                settings[key] = cfg_user.training_settings[key]
            else:
                settings[key] = cfg_default.training_settings[key]
        else:
            if key in cfg_user.general:
                settings[key] = cfg_user.general[key]
            else:
                settings[key] = cfg_default.general[key]

    # Iterate over each dataset configuration and verify whether each key has been defined for the dataset.
    # If not, assign the previously extracted settings to the dataset.
    for i, dataset in enumerate(cfg_user.datasets):
        for key, value in settings.items():
            if key == 'n_training_samples' and 'train' not in dataset.area_type:
                pass
            elif key == 'test_stripe' and dataset.allocation_strategy == 'entire':
                pass
            elif key not in dataset:
                dataset[key] = value


# Verify user config file: dataset settings
def _valid_dataset_args(cfg_user, logger):
    """
    Validates the "datasets" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    if 'datasets' not in cfg_user:
        logger.error("No datasets defined. Please provide a list composed of at least one dictionary to define "
                     "the training and validation dataset(s).")
        logger.info("Mandatory keys of each dictionary:")
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_MANDATORY_train]))
        logger.info("Optional keys of each dictionary:")
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_OPTIONAL]))
        return False
    elif not isinstance(cfg_user.datasets, list) or isinstance(cfg_user.datasets, list) and len(cfg_user.datasets) == 0:
        logger.error("Invalid 'datasets' argument. Please provide a list composed of at least one dictionary to define "
                     "the training and validation dataset(s).")
        logger.info("Mandatory keys of each dictionary:")
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_MANDATORY_train]))
        logger.info("Optional keys of each dictionary:")
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_OPTIONAL]))
        return False

    # Extract the input channel configuration (use the default setting if not specified by the user)
    if 'model' in cfg_user and 'input_channels' in cfg_user.model:
        # Use user setting
        input_config = cfg_user.model.input_channels
    else:
        # Use default setting
        input_config = cfg_default.model.input_channels

    # Initialize a list to save which of the datasets is improperly defined (wrong paths, missing arguments, etc.)
    error = [False] * len(cfg_user.datasets)

    # Initialize the number of cross-validation datasets
    n_crossval_datasets = 0

    for i, dataset in enumerate(cfg_user.datasets):
        utils.print_dataset_name_to_console(dataset, i, logger)

        if not all_keys_known(dataset, arguments.DATASET_KEYS_MANDATORY_train + arguments.DATASET_KEYS_OPTIONAL,
                              logger):
            error[i] = True

        # Verify that the initial depth/height raster (initial DSM) exists
        if 'raster_in' not in dataset:
            logger.error("Missing argument 'raster_in'. Specify the path of the initial depth/height raster "
                         "(initial DSM).\n")
            error[i] = True
        elif not is_string(dataset.raster_in, 'raster_in', logger):
            error[i] = True
        elif not fdutil.file_exists(dataset.raster_in):
            logger.error(f"Initial depth/height raster (initial DSM) does not exist: {dataset.raster_in}\n")
            error[i] = True

        # Verify that the ground truth depth/height raster (ground truth DSM) exists
        if 'raster_gt' not in dataset:
            logger.error("Missing argument 'raster_gt'. Specify the path of the ground truth depth/height raster "
                         "(ground truth DSM).\n")
            error[i] = True
        elif not is_string(dataset.raster_gt, 'raster_gt', logger):
            error[i] = True
        elif not fdutil.file_exists(dataset.raster_gt):
            logger.error(f"Ground truth depth/height raster (ground truth DSM) does not exist: {dataset.raster_gt}\n")
            error[i] = True

        # True if ResDepth is trained with image guidance
        if input_config in ['geom-multiview', 'geom-stereo', 'geom-mono', 'stereo']:
            # Verify that the image list exists
            if 'path_image_list' not in dataset:
                logger.error("Missing argument 'path_image_list'. Specify a text file which stores the paths of "
                             "the ortho-rectified images.\n")
                error[i] = True
            elif not is_string(dataset.path_image_list, 'path_image_list', logger):
                error[i] = True
            elif fdutil.file_extension(dataset.path_image_list) != '.txt':
                logger.error("Invalid value for the argument 'path_image_list'. Specify a text file which stores the "
                             "paths of the ortho-rectified images.\n")
                error[i] = True
            elif not fdutil.file_exists(dataset.path_image_list):
                logger.error(f"Cannot find the image list: {dataset.path_image_list}\n")
                error[i] = True
            else:
                # Verify that each image listed in the image list exists
                image_paths = io_control_file.read_imagelist_from_file(dataset.path_image_list)
                for path in image_paths:
                    if not fdutil.file_exists(path):
                        logger.error(f"The following image (listed in 'path_image_list') does not exist: {path}\n")
                        error[i] = True

            if 'area_type' in dataset and isinstance(dataset.area_type, str):
                if 'train' in dataset.area_type:
                    if 'path_pairlist_training' not in dataset:
                        logger.error("Missing argument 'path_pairlist_training'. Specify a text file which stores "
                                     "the image pairs used for training.\n")
                        error[i] = True
                    elif not is_string(dataset.path_pairlist_training, 'path_pairlist_training', logger):
                        error[i] = True
                    elif fdutil.file_extension(dataset.path_pairlist_training) != '.txt':
                        logger.error("Invalid value for the argument 'path_pairlist_training'. Specify a text file "
                                     "that stores the image pairs used for training.\n")
                        error[i] = True
                    elif not fdutil.file_exists(dataset.path_pairlist_training):
                        logger.error("Cannot find the image pair list used for training: "
                                     f"{dataset.path_pairlist_training}\n")
                        error[i] = True
                if 'train' not in dataset.area_type and 'path_pairlist_training' in dataset:
                    logger.warning("This dataset will not be used for training.\n"
                                   f"Hence, the image pair list {dataset.path_pairlist_training} will be ignored.\n")

                if 'val' in dataset.area_type:
                    if 'path_pairlist_validation' not in dataset:
                        logger.error("Missing argument 'path_pairlist_validation'. Specify a text file which stores "
                                     "the image pairs used for validation.\n")
                        error[i] = True
                    elif not is_string(dataset.path_pairlist_validation, 'path_pairlist_validation', logger):
                        error[i] = True
                    elif fdutil.file_extension(dataset.path_pairlist_validation) != '.txt':
                        logger.error("Missing argument 'path_pairlist_validation'. Specify a text file which stores "
                                     "the image pairs used for validation.\n")
                        error[i] = True
                    elif not fdutil.file_exists(dataset.path_pairlist_validation):
                        logger.error("Cannot find the image pair list used for validation: "
                                     f"{dataset.path_pairlist_validation}\n")
                        error[i] = True
                if 'val' not in dataset.area_type and 'path_pairlist_validation' in dataset:
                    logger.warning("This dataset will not be used for validation.\n"
                                   f"Hence, the image pair list {dataset.path_pairlist_validation} will be ignored.\n")

        # True if ResDepth is trained without image guidance
        if input_config == 'geom':
            if 'path_image_list' in dataset or 'path_pairlist_training' in dataset or \
                    'path_pairlist_validation' in dataset:
                logger.warning("The argument 'model': 'input_channels' is set to 'geom', i.e., ResDepth will be "
                               "trained without image guidance. The arguments 'path_image_list', "
                               "'path_pairlist_training', and 'path_pairlist_validation' are ignored.\n"
                               "Set 'model': 'input_channels' to one of the following options to train "
                               "ResDepth using image guidance:\n"
                               "['geom-multiview', 'geom-stereo', 'geom-mono', 'stereo'].\n")

        # Make sure that the dataset is either defined as training or validation dataset (or used for both training and
        # validation if the dataset is split into multiple stripes according to the key "allocation_strategy")
        if 'area_type' not in dataset:
            logger.error("Specify whether this dataset should be used for training or validation. Choose among "
                         f"{arguments.DATASET_AREA_TYPES} to specify 'area_type'.\n")
            error[i] = True
        else:
            if not is_string(dataset.area_type, 'area_type', logger):
                logger.error(f"Choose among {arguments.DATASET_AREA_TYPES} to specify 'area_type'.\n")
                error[i] = True
            else:
                if dataset.area_type not in arguments.DATASET_AREA_TYPES:
                    logger.error(f"Invalid 'area_type' of the dataset: '{dataset.area_type}'. Choose among "
                                 f"{arguments.DATASET_AREA_TYPES} to specify 'area_type'.\n")
                    error[i] = True

        # Verify that the number of training samples is a positive integer
        if 'n_training_samples' in dataset and not is_positive_integer(dataset.n_training_samples,
                                                                       'n_training_samples', logger):
            error[i] = True

        # Verify that the allocation strategy (train/val/test split) is valid
        if 'allocation_strategy' in dataset:
            allocation = dataset.allocation_strategy
            if not is_string(allocation, 'allocation_strategy', logger):
                error[i] = True
            elif not valid_allocation(allocation, logger):
                error[i] = True

        elif 'allocation_strategy' in cfg_user.general:
            allocation = cfg_user.general.allocation_strategy
            if not is_string(allocation, 'allocation_strategy (general settings)', logger):
                error[i] = True
            elif not valid_allocation(allocation, logger):
                error[i] = True
        else:
            allocation = cfg_default.general.allocation_strategy

        if allocation == '5-crossval_vertical' or allocation == '5-crossval_horizontal':
            if 'test_stripe' in dataset:
                if not is_positive_integer(dataset.test_stripe, 'test_stripe', logger, zero_allowed=True):
                    error[i] = True
                elif dataset.test_stripe > 4:
                    logger.error(f"Invalid number of stripes for the allocation strategy '{allocation}'. Set "
                                 "'test_stripe' to one of the following integers [0, 1, 2, 3, 4].\n")
                    error[i] = True

            elif 'test_stripe' in cfg_user.general:
                if not is_positive_integer(cfg_user.general.test_stripe, "general': 'test_stripe",
                                           logger, zero_allowed=True):
                    error[i] = True
                elif cfg_user.general.test_stripe > 4:
                    logger.error(f"Invalid number of stripes for the allocation strategy '{allocation}' "
                                 "(general settings). Set 'test_stripe' to one of the following integers "
                                 "[0, 1, 2, 3, 4].\n")
                    error[i] = True

        if allocation == 'entire' and 'area_type' in dataset and '+' in dataset.area_type:
            # True if allocation='entire' and dataset.area_type='train+val':
            logger.error(f"'area_type' cannot be set to '{dataset.area_type}' in conjunction with "
                         "'allocation_strategy': 'entire'. Either choose among ['train', 'val'] to specify 'area_type' "
                         "or choose among ['5-crossval_vertical', '5-crossval_horizontal'] to define "
                         "'allocation_strategy'.\n")
            error[i] = True

        # Verify the flag to activate/deactivate cross-validation
        if 'crossval_training' in dataset:
            if not is_boolean(dataset.crossval_training, 'crossval_training', logger):
                error[i] = True
            elif dataset.crossval_training:
                n_crossval_datasets += 1

        if error[i]:
            logger.info('\n')

    error = any(error)

    # Make sure that the configuration file specifies at least one training and one validation dataset,
    # or alternatively, one single dataset with training and validation (and testing) split
    for key in ['train', 'val']:
        count = 0
        for i, dataset in enumerate(cfg_user.datasets):
            if 'area_type' in dataset and isinstance(dataset.area_type, str) and key in dataset.area_type:
                count += 1

        if count == 0 and key == 'train':
            logger.error(f"Specify at least one training dataset! Choose among {arguments.DATASET_AREA_TYPES} to "
                         "specify 'datasets': 'area_type' or specify an additional dictionary to define a "
                         "training dataset.\n")
            error = True
        elif count == 0 and key == 'val':
            logger.error(f"Specify at least one validation dataset! Choose among {arguments.DATASET_AREA_TYPES} "
                         "to specify 'datasets': 'area_type' or specify an additional dictionary to define a "
                         "validation dataset.\n")
            error = True

    # Make sure that the configuration file specifies exactly one dataset if cross-validation is activated
    if n_crossval_datasets > 1 or (n_crossval_datasets == 1 and len(cfg_user.datasets) > 1):
        logger.error(f"Specify one dataset only to perform cross-validation. Set 'area_type' to 'train+val' and choose "
                     f"among ['5-crossval_vertical', '5-crossval_horizontal'] to specify 'allocation_strategy'.\n")
        error = True

    # Cross-validation: Make sure that the allocation strategy is valid
    if n_crossval_datasets == 1 and len(cfg_user.datasets) == 1 and 'allocation_strategy' in cfg_user.datasets[0] and \
            cfg_user.datasets[0].allocation_strategy == 'entire':
        logger.error(f"Invalid allocation strategy '{cfg_user.datasets[0].allocation_strategy}'. Choose among "
                     "['5-crossval_vertical', '5-crossval_horizontal'] to perform cross-validation.\n")
        error = True

    if not error:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: model settings
def _valid_model_args(cfg_user, logger):
    """
    Validates the "model" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    if 'model' in cfg_user:
        if not all_keys_known(cfg_user.model, arguments.MODEL_KEYS, logger):
            error = True

        # Verify the input channel configuration
        if 'input_channels' in cfg_user.model and cfg_user.model.input_channels not in arguments.INPUT_CHANNELS:
            logger.error(f"Unknown input channel configuration '{cfg_user.model.input_channels}'. Choose among "
                         f"{arguments.INPUT_CHANNELS} to specify 'input_channels'.\n")
            error = True

        # Verify the network architecture
        if 'name' in cfg_user.model and cfg_user.model.name not in arguments.ARCHITECTURES:
            logger.error(f"The specified model architecture '{cfg_user.model.name}' is not implemented. Choose among "
                         f"{arguments.ARCHITECTURES} to specify 'name'.\n")
            error = True

        name = cfg_user.model.name if 'name' in cfg_user.model else cfg_default.model.name

        if name == 'UNet':
            # Verify the depth of the UNet
            if 'depth' in cfg_user.model:
                if not is_positive_integer(cfg_user.model.depth, 'depth', logger, zero_allowed=False):
                    error = True

            # Verify the number of filters of the first convolutional encoder layer
            if 'start_kernel' in cfg_user.model:
                if not is_positive_integer(cfg_user.model.start_kernel, 'start_kernel', logger, zero_allowed=False):
                    error = True

            # Verify the activation functions
            if 'act_fn_encoder' in cfg_user.model and not valid_act_fn(cfg_user.model.act_fn_encoder, 'encoder',
                                                                       "'act_fn_encoder'", logger):
                error = True

            if 'act_fn_decoder' in cfg_user.model and not valid_act_fn(cfg_user.model.act_fn_decoder, 'decoder',
                                                                       "'act_fn_decoder'", logger):
                error = True

            if 'act_fn_bottleneck' in cfg_user.model and not valid_act_fn(cfg_user.model.act_fn_bottleneck,
                                                                          'bottleneck', "''act_fn_bottleneck'", logger):
                error = True

            # Verify the up-convolution
            if 'up_mode' in cfg_user.model and cfg_user.model.up_mode not in arguments.UPSAMPLING_MODES:
                logger.error(f"Invalid upsampling layer: '{cfg_user.model.up_mode}'. Choose among "
                             f"{arguments.UPSAMPLING_MODES} to specify 'up_mode'.\n")
                error = True

            # Verify the flag to activate/deactivate the batch normalization layers
            if 'do_BN' in cfg_user.model and not is_boolean(cfg_user.model.do_BN, 'do_BN', logger):
                error = True

            # Verify the flag to activate/deactivate the long residual skip connection that adds the initial DSM
            # to the output of the last decoder layer
            if 'outer_skip' in cfg_user.model and not is_boolean(cfg_user.model.outer_skip, 'outer_skip', logger):
                error = True

            # Verify the flag to add/skip batch normalization to the long residual skip connection
            if 'outer_skip_BN' in cfg_user.model and not is_boolean(cfg_user.model.outer_skip_BN,
                                                                    'outer_skip_BN', logger):
                error = True

            # Verify the flag to activate/deactivate the bias of the convolutional layers
            # (default: switched off if convolutional layers are followed by batch normalization)
            if 'bias_conv_layer' in cfg_user.model and not is_boolean(cfg_user.model.bias_conv_layer,
                                                                      'bias_conv_layer', logger):
                error = True

        # If the depth/height raster is omitted in the network input, the long residual skip connection
        # cannot be applied:
        if 'input_channels' in cfg_user.model and cfg_user.model.input_channels == 'stereo':
            outer_skip = cfg_user.model.outer_skip if 'outer_skip' in cfg_user.model else cfg_default.model.outer_skip

            if outer_skip:
                logger.warning("Cannot apply the long residual skip connection when using ortho-rectified stereo "
                               "images as the sole input to ResDepth. Either deactivate the long residual skip "
                               "connection by setting\n"
                               "'model': 'outer_skip' = False\n"
                               "or use a depth/height map as an additional network input by specifying\n"
                               "'model': 'input_channels' = 'geom-stereo'.\n")
                error = True

        # Verify that the pretrained model weights exist
        if 'pretrained_path' in cfg_user.model:
            if not is_string(cfg_user.model.pretrained_path, 'pretrained_path', logger):
                error = True
            elif not fdutil.file_exists(cfg_user.model.pretrained_path):
                logger.error(f"Cannot find the pretrained model weights: {cfg_user.model.pretrained_path}\n")
                error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: multiview settings
def _valid_multiview_args(cfg_user, logger):
    """
    Validates the "multiview" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    # Extract the input channel configuration (use the default setting if not specified by the user)
    if 'model' in cfg_user and 'input_channels' in cfg_user.model:
        # Use user setting
        input_config = cfg_user.model.input_channels
    else:
        # Use default setting
        input_config = cfg_default.model.input_channels

    if input_config != 'geom-multiview' and 'multiview' in cfg_user:
        logger.warning(f"The argument 'model': 'input_channels' is set to '{input_config}'. Hence, the multiview "
                       "settings will be ignored.\n")

    elif input_config == 'geom-multiview' and 'multiview' in cfg_user:
        if not all_keys_known(cfg_user.multiview, arguments.MULTIVIEW_KEYS, logger):
            error = True

        if 'config' in cfg_user.multiview and cfg_user.multiview.config not in arguments.MULTIVIEW_CONFIG:
            logger.error(f"Unknown multiview configuration: '{cfg_user.multiview.config}'. Choose among "
                         f"{arguments.MULTIVIEW_CONFIG} to specify 'config'.\n")
            error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: stereo pair settings
def _valid_stereo_args(cfg_user, logger):
    """
    Validates the "stereopair_settings" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    # Extract the input channel configuration (use the default setting if not specified by the user)
    if 'model' in cfg_user and 'input_channels' in cfg_user.model:
        # Use user setting
        input_config = cfg_user.model.input_channels
    else:
        # Use default setting
        input_config = cfg_default.model.input_channels

    if input_config in ['geom', 'geom-mono'] and 'stereopair_settings' in cfg_user:
        logger.warning(f"The argument 'model': 'input_channels' is set to '{input_config}'. Hence, the stereo pair "
                       "settings will be ignored.\n")

    elif 'stereopair_settings' in cfg_user:
        if not all_keys_known(cfg_user.stereopair_settings, arguments.STEREO_KEYS, logger):
            error = True

        if 'use_all_stereo_pairs' in cfg_user.stereopair_settings:
            if not is_boolean(cfg_user.stereopair_settings.use_all_stereo_pairs, 'use_all_stereo_pairs', logger):
                error = True

        if 'permute_images_within_pair' in cfg_user.stereopair_settings:
            if not is_boolean(cfg_user.stereopair_settings.permute_images_within_pair, 'permute_images_within_pair',
                              logger):
                error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: training settings
def _valid_training_args(cfg_user, logger):
    """
    Validates the "training_settings" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    if 'training_settings' in cfg_user:
        if not all_keys_known(cfg_user.training_settings, arguments.TRAINING_KEYS, logger):
            error = True

        # Verify that the number of training samples is a positive integer
        if 'n_training_samples' in cfg_user.training_settings and not \
                is_positive_integer(cfg_user.training_settings.n_training_samples, 'n_training_samples', logger):
            error = True

        # Verify the tile size
        if 'model' in cfg_user and 'depth' in cfg_user.model:
            min_power = cfg_user.model.depth + 2   # Consistency with the number of downsampling layers
        else:
            min_power = cfg_default.model.depth + 2
        if 'tile_size' in cfg_user.training_settings and not valid_tile_size(cfg_user.training_settings.tile_size,
                                                                             'tile_size', min_power, logger):
            error = True

        # Verify that 'augment' is a boolean
        if 'augment' in cfg_user.training_settings and not is_boolean(cfg_user.training_settings.augment,
                                                                      'augment', logger):
            error = True

        # Verify the loss function
        if 'loss' in cfg_user.training_settings and cfg_user.training_settings.loss not in arguments.LOSSES:
            logger.error(f"Unknown loss function '{cfg_user.training_settings.loss}'. Choose among "
                         f"{arguments.LOSSES} to specify 'loss'.\n")
            error = True

        # Verify that the batch size is a positive integer
        if 'batch_size' in cfg_user.training_settings and not is_positive_integer(cfg_user.training_settings.batch_size,
                                                                                  'batch_size', logger):
            error = True

        # Verify that the number of training epochs is a positive integer
        if 'n_epochs' in cfg_user.training_settings and not is_positive_integer(cfg_user.training_settings.n_epochs,
                                                                                'n_epochs', logger):
            error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: optimizer settings
def _valid_optimizer_args(cfg_user, logger):
    """
    Validates the "optimizer" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    if 'optimizer' in cfg_user:
        if not all_keys_known(cfg_user.optimizer, arguments.OPTIMIZER_KEYS, logger):
            error = True

        if 'name' not in cfg_user.optimizer:
            logger.error(f"The optimizer is not specified. Choose among {arguments.OPTIMIZERS} to specify 'name'.\n")
            error = True
        else:
            if cfg_user.optimizer.name not in arguments.OPTIMIZERS:
                logger.error(f"Unknown optimizer '{cfg_user.optimizer.name}'. Choose among {arguments.OPTIMIZERS} "
                             "to specify 'name'.\n")
                error = True

            if 'learning_rate' in cfg_user.optimizer and cfg_user.optimizer.learning_rate <= 0:
                logger.error("Invalid value for the argument 'learning_rate': "
                             f"{cfg_user.optimizer.learning_rate}. Specify a positive number.\n")
                error = True

            if 'weight_decay' in cfg_user.optimizer and cfg_user.optimizer.weight_decay <= 0:
                logger.error("Invalid value for the argument 'weight_decay': "
                             f"{cfg_user.optimizer.weight_decay}. Specify a positive number.\n")
                error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: scheduler settings
def _valid_scheduler_args(cfg_user, logger):
    """
    Validates the "scheduler" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    if 'scheduler' in cfg_user:
        if not all_keys_known(cfg_user.scheduler, arguments.SCHEDULER_KEYS, logger):
            error = True

        if 'name' not in cfg_user.scheduler:
            logger.error(f"The learning rate scheduler is not specified. Choose among {arguments.SCHEDULERS} to "
                         "specify 'name'.\n")
            error = True
        else:
            if cfg_user.scheduler.name not in arguments.SCHEDULERS:
                logger.error(f"Unknown scheduler '{cfg_user.scheduler.name}'. Choose among {arguments.SCHEDULERS} "
                             "to specify 'name'.\n")
                error = True

            if 'enabled' in cfg_user.scheduler:
                if not is_boolean(cfg_user.scheduler.enabled, 'enabled', logger):
                    error = True
            else:
                logger.error("Missing argument 'enabled'. Enable or disable the learning rate scheduler.\n")
                error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: general settings
def _valid_general_args(cfg_user, logger):
    """
    Validates the "general" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    if 'general' in cfg_user:
        if not all_keys_known(cfg_user.general, arguments.GENERAL_KEYS, logger):
            error = True

        # Verify that the allocation strategy (train/val/test split) is valid
        if 'allocation_strategy' in cfg_user.general:
            allocation = cfg_user.general.allocation_strategy

            if not is_string(allocation, 'allocation_strategy', logger):
                error = True
            elif not valid_allocation(allocation, logger):
                error = True
        else:
            allocation = cfg_default.general.allocation_strategy

        if allocation == '5-crossval_vertical' or allocation == '5-crossval_horizontal':
            if 'test_stripe' in cfg_user.general:
                if not is_positive_integer(cfg_user.general.test_stripe, 'test_stripe', logger, zero_allowed=True):
                    error = True
                elif cfg_user.general.test_stripe > 4:
                    logger.error(f"Invalid number of stripes for the allocation strategy '{allocation}'. Set "
                                 "'test_stripe' to one of the following integers [0, 1, 2, 3, 4].\n")
                    error = True

        # Verify the number of workers
        if 'workers' in cfg_user.general:
            if not is_positive_integer(cfg_user.general.workers, 'workers', logger, zero_allowed=True):
                error = True
            if multiprocessing.cpu_count() < cfg_user.general.workers:
                logger.error(f"Requested to use {cfg_user.general.workers} cores, but only "
                             f"{multiprocessing.cpu_count()} cores are available.\n")
                error = True

        # Verify the random seed
        if 'random_seed' in cfg_user.general:
            if not is_positive_integer(cfg_user.general.random_seed, 'random_seed', logger, zero_allowed=True):
                error = True

        # Verify that the frequency of model logging is a positive integer
        if 'save_model_rate' in cfg_user.general and not is_positive_integer(cfg_user.general.save_model_rate,
                                                                             'save_model_rate', logger):
            error = True

        # Verify that the frequency of model evaluation is a positive integer
        if 'evaluate_rate' in cfg_user.general and not is_positive_integer(cfg_user.general.evaluate_rate,
                                                                           'evaluate_rate', logger):
            error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


# Verify user config file: output settings
def _valid_output_args(cfg_user, logger):
    """
    Validates the "output" parameters of a json configuration file used for training.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    if not all_keys_known(cfg_user.output, arguments.OUTPUT_KEYS, logger):
        error = True

    # Check if the result and the logs directories exist, and if not, create directories
    if 'output_directory' not in cfg_user.output:
        logger.error("Missing argument 'output_directory'. Specify the output directory (directory where the results "
                     "will be stored).\n")
        error = True
    elif not is_string(cfg_user.output.output_directory, 'output_directory', logger):
        error = True
    else:
        fdutil.make_dir(cfg_user.output.output_directory)

    if 'tboard_log_dir' not in cfg_user.output:
        logger.error("Missing argument 'tboard_log_dir'. Specify the logs directory (directory where the tensorboard "
                     "checkpoint files [event files] will be stored).\n")
        error = True
    elif not is_string(cfg_user.output.tboard_log_dir, 'tboard_log_dir', logger):
        error = True
    else:
        fdutil.make_dir(cfg_user.output.tboard_log_dir)

    # Verify the suffix of the output directory
    if 'suffix' in cfg_user.output and not is_string(cfg_user.output.suffix, 'suffix', logger):
        error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error
