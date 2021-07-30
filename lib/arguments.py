# ---------------------------------- Primary keys (Training) ---------------------------------- #

PRIMARY_KEYS = ['datasets', 'model', 'multiview', 'stereopair_settings', 'training_settings', 'optimizer',
                'scheduler', 'general', 'output']

PRIMARY_KEYS_MANDATORY = ['datasets', 'output']

# --------------------------------- Secondary keys (Training) --------------------------------- #

DATASET_KEYS_MANDATORY_train = ['raster_gt', 'raster_in', 'area_type']
DATASET_KEYS_OPTIONAL = ['name', 'path_image_list', 'path_pairlist_training', 'path_pairlist_validation',
                         'n_training_samples', 'allocation_strategy', 'test_stripe', 'crossval_training']

MODEL_KEYS = ['name', 'input_channels', 'depth', 'start_kernel', 'act_fn_encoder', 'act_fn_decoder',
              'act_fn_bottleneck', 'up_mode', 'do_BN', 'bias_conv_layer', 'outer_skip', 'outer_skip_BN',
              'pretrained_path']

MULTIVIEW_KEYS = ['config']

STEREO_KEYS = ['use_all_stereo_pairs', 'permute_images_within_pair']

TRAINING_KEYS = ['n_training_samples', 'tile_size', 'augment', 'loss', 'batch_size', 'n_epochs']

OPTIMIZER_KEYS = ['name', 'learning_rate', 'weight_decay']

SCHEDULER_KEYS = ['enabled', 'name', 'settings']

GENERAL_KEYS = ['allocation_strategy', 'test_stripe', 'workers', 'random_seed', 'save_model_rate', 'evaluate_rate']

OUTPUT_KEYS = ['output_directory', 'tboard_log_dir', 'suffix', 'plot_model_txt']


# --------------------------------- Primary keys (Inference) --------------------------------- #

PRIMARY_KEYS_eval = ['datasets', 'model', 'general', 'output']

# -------------------------------- Secondary keys (Inference) -------------------------------- #

DATASET_KEYS_MANDATORY_eval = ['raster_in']
DATASET_KEYS_OPTIONAL_eval = ['name', 'raster_gt', 'path_image_list', 'path_pairlist', 'mask_ground_truth',
                              'mask_building', 'mask_water', 'mask_forest', 'allocation_strategy', 'test_stripe',
                              'area_type', 'crossval_training']
MODEL_KEYS_eval = ['weights', 'architecture', 'normalization_geom', 'normalization_image']
GENERAL_KEYS_eval = ['tile_size', 'workers']


# --------------------------------------- Valid values --------------------------------------- #

DATASET_AREA_TYPES = ['train', 'val', 'train+val']
DATASET_AREA_TYPES_eval = ['train', 'val', 'test']
INPUT_CHANNELS = ['geom-multiview', 'geom-stereo', 'geom-mono', 'stereo', 'geom']
MULTIVIEW_CONFIG = ['3-view', '4-view', '5-view']
OPTIMIZERS = ['Adam', 'SGD']
SCHEDULERS = ['ReduceLROnPlateau', 'StepLR', 'ExponentialLR']
LOSSES = ['L1']
ARCHITECTURES = ['UNet']
ACTIVATION_FUNCTIONS = ['relu', 'lrelu', 'prelu']
UPSAMPLING_MODES = ['transpose', 'bilinear']
ALLOCATION_STRATEGIES = ['5-crossval_vertical', '5-crossval_horizontal', 'entire']
