from copy import deepcopy
from easydict import EasyDict as edict
import multiprocessing
import sys

from lib import arguments, cfg_utils, fdutil, io_control_file, utils
from lib.config import cfg as cfg_default
from lib.validate_arguments import all_keys_known, is_boolean, is_string, is_positive_integer, valid_allocation,\
    valid_tile_size


def validate_and_update_cfg_file(cfg_file, logger=None):
    """
    Validates a json configuration file used for inference. Specifically, the function
    a) detects unknown keys.
    b) checks that all required keys are defined.
    c) checks that all values are valid.
    d) checks that all input data exists.

    :param cfg_file:  str, Dict or EasyDict; path of the json file or file imported as dictionary returned by
                      the function cfg_utils.read_json()
    :param logger:    logger instance (if None, output is print to console)
    :return:          EasyDict, dictionary composed the following keys:
                      status:   bool, True if no errors have been detected in the configuration file, False otherwise
                      cfg:      EasyDict, updated json configuration file
    """

    if logger is None:
        logger = utils.setup_logger('validate_cfg_file', log_to_console=True, log_file=None)

    # Load the configuration file
    if isinstance(cfg_file, dict):
        cfg = edict(deepcopy(cfg_file))
    elif isinstance(cfg_file, edict):
        cfg = deepcopy(cfg_file)
    else:
        cfg = cfg_utils.read_json(cfg_file)
        if not cfg:
            sys.exit(1)

    # Verify whether all primary keys are known
    if not all_keys_known(cfg, arguments.PRIMARY_KEYS_eval, logger):
        return edict({'status': False, 'cfg': {}})

    # Verify that all primary keys are specified
    missing_keys = [k for k in arguments.PRIMARY_KEYS_eval if k not in cfg]
    if len(missing_keys) > 0:
        logger.error('The following keys are missing: {}.\n'.format(','.join(["'{}'".format(x) for x in missing_keys])))
        return edict({'status': False, 'cfg': {}})

    # Verify user config file: model settings
    process = "Verify 'model' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if _valid_model_args(cfg, logger) is False:
        return edict({'status': False, 'cfg': {}})

    # Read model architecture and settings
    cfg.model.update(cfg_utils.read_json(cfg.model.architecture))

    # Verify user config file: dataset settings
    process = "Verify 'datasets' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))
    eval_cfg = _valid_dataset_args(cfg, cfg.model.input_channels, logger)

    if eval_cfg.status is False:
        return edict({'status': False, 'cfg': {}})
    else:
        cfg = eval_cfg.cfg
        del eval_cfg

    # Verify user config file: general settings
    process = "Verify 'general' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))
    eval_cfg = _valid_general_args(cfg, logger)

    if eval_cfg.status is False:
        return edict({'status': False, 'cfg': {}})
    else:
        cfg = eval_cfg.cfg
        del eval_cfg

    # Verify user config file: output settings
    process = "Verify 'output' arguments"
    logger.info('{}\n{}\n'.format(process, '-' * len(process)))

    if _valid_output_args(cfg, logger) is False:
        return edict({'status': False, 'cfg': {}})

    return edict({'status': True, 'cfg': cfg})


def _valid_dataset_args(cfg, input_config, logger):
    """
    Validates the "datasets" parameters of a json configuration file used for inference.

    :param cfg:             EasyDict, json configuration file imported as dictionary
    :param input_config:    str, input channel configuration (model architecture)
    :param logger:          logger instance
    :return:                EasyDict, dictionary composed the following keys:
                            status: bool, True if no errors have been detected in the configuration file,
                                    False otherwise
                            cfg:    EasyDict, updated json configuration file
    """

    if 'datasets' not in cfg:
        logger.error('No datasets defined. Please provide a list composed of at least one dictionary to define the '
                     'input data. Specify at least one dictionary with the following keys:')
        logger.info('Mandatory keys:')
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_MANDATORY_eval]))
        logger.info('Additional optional keys:')
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_OPTIONAL_eval]))
        return edict({'status': False, 'cfg': {}})
    elif not isinstance(cfg.datasets, list) or isinstance(cfg.datasets, list) and len(cfg.datasets) == 0:
        logger.error("Invalid 'datasets' argument. Please provide a list composed of at least one dictionary to define "
                     "the input data. Specify at least one dictionary with the following keys:")
        logger.info('Mandatory keys:')
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_MANDATORY_eval]))
        logger.info('Additional optional keys:')
        logger.info(', '.join(["'{0}'".format(x) for x in arguments.DATASET_KEYS_OPTIONAL_eval]))
        return edict({'status': False, 'cfg': {}})

    # Initialize a list to save which of the datasets is improperly defined (wrong paths, missing arguments, etc.)
    error = [False] * len(cfg.datasets)

    for i, dataset in enumerate(cfg.datasets):
        utils.print_dataset_name_to_console(dataset, i)

        # Verify that all keys are known
        if not all_keys_known(dataset, arguments.DATASET_KEYS_MANDATORY_eval + arguments.DATASET_KEYS_OPTIONAL_eval,
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
            logger.error(f"Initial depth/height raster (initial DSM) does not exist:\n{dataset.raster_in}\n")
            error[i] = True

        # Verify that the ground truth depth/height raster (ground truth DSM) exists
        if 'raster_gt' in dataset:
            if not is_string(dataset.raster_gt, 'raster_gt', logger):
                error[i] = True
            elif not fdutil.file_exists(dataset.raster_gt):
                logger.error(f"Ground truth depth/height raster (ground truth DSM) does not exist:"
                             f"\n{dataset.raster_gt}\n")
                error[i] = True

        # Verify that the ground truth mask raster exists (if specified)
        if 'mask_ground_truth' in dataset:
            if not is_string(dataset.mask_ground_truth, 'mask_ground_truth', logger):
                error[i] = True
            elif not fdutil.file_exists(dataset.mask_ground_truth):
                logger.error(f"Ground truth mask raster does not exist:\n{dataset.mask_ground_truth}\n")
                error[i] = True

        # Verify that the building mask raster exists (if specified)
        if 'mask_building' in dataset:
            if not is_string(dataset.mask_building, 'mask_building', logger):
                error[i] = True
            elif not fdutil.file_exists(dataset.mask_building):
                logger.error(f"Building mask raster does not exist:\n{dataset.mask_building}\n")
                error[i] = True

        # Verify that the water mask raster exists (if specified)
        if 'mask_water' in dataset:
            if not is_string(dataset.mask_water, 'mask_water', logger):
                error[i] = True
            elif not fdutil.file_exists(dataset.mask_water):
                logger.error(f"Water mask raster does not exist:\n{dataset.mask_water}\n")
                error[i] = True

        # Verify that the forest mask raster exists (if specified)
        if 'mask_forest' in dataset:
            if not is_string(dataset.mask_forest, 'mask_forest', logger):
                error[i] = True
            elif not fdutil.file_exists(dataset.mask_forest):
                logger.error(f"Forest mask raster does not exist:\n{dataset.mask_forest}\n")
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
                logger.error("Wrong argument 'path_image_list'. Specify a text file which stores the paths of "
                             "the ortho-rectified images.\n")
                error[i] = True
            elif not fdutil.file_exists(dataset.path_image_list):
                logger.error(f"Cannot find the image list:\n{dataset.path_image_list}\n")
                error[i] = True
            else:
                # Verify that each image listed in the image list exists
                image_paths = io_control_file.read_imagelist_from_file(dataset.path_image_list)
                for path in image_paths:
                    if not fdutil.file_exists(path):
                        logger.error(f"The following image does not exist:\t{path}\n")
                        error[i] = True

            if 'path_pairlist' not in dataset:
                logger.error("Missing argument 'path_pairlist'. Specify a text file which stores the image pairs used "
                             "for inference.\n")
                error[i] = True
            elif not is_string(dataset.path_pairlist, 'path_pairlist', logger):
                error[i] = True
            elif fdutil.file_extension(dataset.path_pairlist) != '.txt':
                logger.error("Wrong argument 'path_pairlist'. Specify a text file which stores the image pairs used "
                             "for inference.\n")
                error[i] = True
            elif not fdutil.file_exists(dataset.path_pairlist):
                logger.error(f"Cannot find the image pair list used for inference:\n{dataset.path_pairlist}\n")
                error[i] = True

        # True if ResDepth is trained without image guidance
        if input_config == 'geom':
            if 'path_image_list' in dataset or 'path_pairlist' in dataset:
                logger.error("The specified ResDepth model is trained without image guidance.\n"
                             "The arguments 'path_image_list' and 'path_pairlist' are thus ignored.\n")

        # Verify that the allocation strategy is valid
        if 'allocation_strategy' in dataset:
            if not is_string(dataset.allocation_strategy, 'allocation_strategy', logger):
                error[i] = True
            elif not valid_allocation(dataset.allocation_strategy, logger):
                error[i] = True

        elif 'allocation_strategy' in cfg.general:
            dataset.allocation_strategy = cfg.general.allocation_strategy
        else:
            # Refine the entire raster
            dataset.allocation_strategy = 'entire'

        if dataset.allocation_strategy == '5-crossval_vertical' or \
                dataset.allocation_strategy == '5-crossval_horizontal':
            if 'test_stripe' not in dataset:
                logger.error(f"Missing argument 'test_stripe'. Specify which of the 5 stripes should be used for "
                             "inference by setting 'test_stripe' to one of the following integers [0, 1, 2, 3, 4].\n")
                error[i] = True
            else:
                if not is_positive_integer(dataset.test_stripe, 'test_stripe', logger, zero_allowed=True):
                    error[i] = True
                elif dataset.test_stripe > 4:
                    logger.error(f"Invalid number of stripes for the allocation strategy "
                                 f"'{dataset.allocation_strategy}'. Set 'test_stripe' to one of the following "
                                 "integers [0, 1, 2, 3, 4].\n")
                    error[i] = True

            if 'area_type' not in dataset:
                logger.error(f"Missing argument 'area_type' for the allocation strategy "
                             f"'{dataset.allocation_strategy}'. Specify which area should be used for inference. "
                             f"Choose among {arguments.DATASET_AREA_TYPES_eval}.\n")
                error[i] = True
            else:
                if not is_string(dataset.area_type, 'area_type', logger):
                    error[i] = True
                elif dataset.area_type not in arguments.DATASET_AREA_TYPES_eval:
                    logger.error(f"Invalid 'area_type' of the dataset: '{dataset.area_type}'. "
                                 f"Choose among {arguments.DATASET_AREA_TYPES_eval} to specify 'area_type'.\n")
                    error[i] = True

            if 'crossval_training' in dataset:
                if not is_boolean(dataset.crossval_training, 'crossval_training', logger):
                    error[i] = True

        if error[i]:
            logger.info('\n')
        else:
            logger.info('Settings check: ok.\n\n')

    error = any(error)

    return edict({'status': not error, 'cfg': cfg})


def _valid_model_args(cfg, logger):
    """
    Validates the "model" parameters of a json configuration file used for inference.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg:     EasyDict, json configuration file imported as dictionary
    :param logger:  logger instance
    :return:        boolean, True if no errors have been detected, False otherwise
    """

    error = False

    if not all_keys_known(cfg.model, arguments.MODEL_KEYS_eval, logger):
        return False

    # Verify the model weights
    if 'weights' not in cfg.model:
        logger.error("Missing argument 'weights'. Specify the path of a pth file which stores the model weights.\n")
        error = True
    elif not is_string(cfg.model.weights, 'weights', logger):
        error = True
    elif fdutil.file_extension(cfg.model.weights) != '.pth':
        logger.error("Wrong argument 'weights'. Specify the path of a pth file which stores the model weights.\n")
        error = True
    elif not fdutil.file_exists(cfg.model.weights):
        logger.error(f"Cannot find the model weights:\n{cfg.model.weights}\n")
        error = True

    # Verify the model architecture
    input_channels = None

    if 'architecture' not in cfg.model:
        logger.error("Missing argument 'architecture'. Specify the path of the file 'model_config.json' "
                     "(stores the model architecture settings; output of train.py).\n")
        error = True
    elif not is_string(cfg.model.architecture, 'architecture', logger):
        error = True
    elif fdutil.file_extension(cfg.model.architecture) != '.json':
        logger.error("Wrong argument 'architecture'. Specify the json file which stores the model architecture "
                     "settings.\n")
        error = True
    elif not fdutil.file_exists(cfg.model.architecture):
        logger.error(f"Cannot find the model architecture settings:\n{cfg.model.architecture}\n")
        error = True
    else:
        cfg_model = cfg_utils.read_json(cfg.model.architecture)
        input_channels = cfg_model.input_channels

    # Verify the depth/height normalization parameters
    if input_channels is not None and input_channels != 'stereo':
        if 'normalization_geom' not in cfg.model:
            logger.error("Missing argument 'normalization_geom'. Specify the path of the pickle file "
                         "'DSM_normalization_parameters.p' (stores the depth/height normalization parameters; output "
                         "of train.py).\n")
            error = True
        elif not is_string(cfg.model.normalization_geom, 'normalization_geom', logger):
            error = True
        elif fdutil.file_extension(cfg.model.normalization_geom) != '.p':
            logger.error("Wrong argument 'normalization_geom'. Specify the absolute path of the pickle file "
                         "'DSM_normalization_parameters.p' which stores the depth/height normalization parameters.\n")
            error = True
        elif not fdutil.file_exists(cfg.model.normalization_geom):
            logger.error(f"Cannot find the depth/height normalization parameters:\n{cfg.model.normalization_geom}\n")
            error = True

    # Verify the image normalization parameters
    if input_channels is not None and input_channels != 'geom':
        if 'normalization_image' not in cfg.model:
            logger.error("Missing argument 'normalization_image'. Specify the path of the pickle file "
                         "'Image_normalization_parameters.p' (stores the image normalization parameters; output "
                         "of train.py).\n")
            error = True
        elif not is_string(cfg.model.normalization_image, 'normalization_image', logger):
            error = True
        elif fdutil.file_extension(cfg.model.normalization_image) != '.p':
            logger.error("Wrong argument 'normalization_image'. Specify the absolute path of the pickle file "
                         "'Image_normalization_parameters.p' which stores the image normalization parameters.\n")
            error = True
        elif not fdutil.file_exists(cfg.model.normalization_image):
            logger.error(f"Cannot find the image normalization parameters:\n{cfg.model.normalization_image}\n")
            error = True

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error


def _valid_general_args(cfg, logger):
    """
    Validates the "general" parameters of a json configuration file used for inference.

    :param cfg:     EasyDict, json configuration file imported as dictionary
    :param logger:  logger instance
    :return:        EasyDict, dictionary composed the following keys:
                    status: bool, True if no errors have been detected in the configuration file, False otherwise
                    cfg:    EasyDict, updated json configuration file
    """

    error = False

    # Verify that all keys are known
    if not all_keys_known(cfg.general, arguments.GENERAL_KEYS_eval, logger):
        error = True

    # Verify that the allocation strategy is valid (global setting)
    if 'allocation_strategy' in cfg.general:
        if not is_string(cfg.general.allocation_strategy, 'allocation_strategy', logger):
            error = True
        elif not valid_allocation(cfg.general.allocation_strategy, logger):
            error = True

    # Verify the tile size
    if 'tile_size' in cfg.general:
        if 'depth' in cfg.model.settings:
            min_power = cfg.model.settings.depth + 2
        else:
            min_power = cfg_default.model.depth + 2

        if not valid_tile_size(cfg.general.tile_size, 'tile_size', min_power, logger):
            error = True
    else:
        cfg.general.tile_size = cfg_default.training_settings.tile_size

    # Verify the number of workers
    if 'workers' in cfg.general:
        if not is_positive_integer(cfg.general.workers, 'workers', logger, zero_allowed=True):
            error = True
        if multiprocessing.cpu_count() < cfg.general.workers:
            logger.error(f"Requested to use {cfg.general.workers} cores, but only {multiprocessing.cpu_count()} "
                         "cores are available.\n")
            error = True
    else:
        cfg.general.workers = multiprocessing.cpu_count()

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return edict({'status': not error, 'cfg': cfg})


def _valid_output_args(cfg_user, logger):
    """
    Validates the "output" parameters of a json configuration file used for inference.
    The function returns False if an error has occurred and True if all settings have passed the check.

    :param cfg_user:  EasyDict, json configuration file imported as dictionary
    :param logger:    logger instance
    :return:          boolean, True if no errors have been detected, False otherwise
    """

    error = False

    # Verify that all keys are known
    if not all_keys_known(cfg_user.output, ['directory'], logger):
        error = True

    if 'directory' not in cfg_user.output:
        logger.error("Missing argument 'directory'. Specify the output directory.\n")
        error = True
    elif not is_string(cfg_user.output.directory, 'directory', logger):
        error = True
    else:
        fdutil.make_dir(cfg_user.output.directory)

    if error:
        logger.info('\n')
    else:
        logger.info('Settings check: ok.\n\n')

    return not error
