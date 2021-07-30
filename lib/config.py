from easydict import EasyDict as edict

# This file defines a dictionary, cfg, which includes the default parameters of the ResDepth pipeline.
# The dictionary is updated/extended at runtime with the parameters defined by the user in the input
# JSON configuration file.

cfg = edict({'model': edict(), 'multiview': edict(), 'stereopair_settings': edict(), 'training_settings': edict(),
             'optimizer': edict(), 'scheduler': edict(), 'general': edict(), 'output': edict()})

# Architecture of the network. Choose among ['UNet'].
cfg.model.name = 'UNet'

# Input channels of the network. Choose among the following configurations:
# ['geom-multiview', 'geom-stereo', 'geom-mono', 'stereo', 'geom'],
# where:
# 'geom':           initial DSM (channel 0) as sole input
# 'stereo':         two ortho-rectified panchromatic stereo views (channel 0 and 1) as input
# 'geom-mono':      initial DSM (channel 0) and one ortho-rectified stereo view (channel 1) as input
# 'geom-stereo':    initial DSM (channel 0) and two ortho-rectified stereo views (channel 1 and 2) as input
# 'geom-multiview': initial DSM (channel 0) and n>2 ortho-rectified stereo views (channels 1,...,n) as input,
#                   where n is specified in cfg.multiview.config; e.g., cfg.multiview.config = '3-view' for n=3
cfg.model.input_channels = 'geom-stereo'

# Number of downsampling and upsampling layers (i.e., number of blocks in the encoder and decoder).
cfg.model.depth = 5

# Activate (True) or deactivate (False) the long residual skip connection that adds the initial DSM to the output of
# the last decoder layer.
cfg.model.outer_skip = True

# Add (True) batch normalization to the long residual skip connection, False otherwise.
cfg.model.outer_skip_BN = False

# Number of filters of the first convolutional layer in the encoder.
cfg.model.start_kernel = 64

# Activation function used in the encoder. Choose among ['relu', 'lrelu', 'prelu'].
cfg.model.act_fn_encoder = 'relu'

# Activation function used in the decoder. Choose among ['relu', 'lrelu', 'prelu'].
cfg.model.act_fn_decoder = 'relu'

# Activation function used in the bottleneck. Choose among ['relu', 'lrelu', 'prelu'].
cfg.model.act_fn_bottleneck = 'relu'

# Upsampling mode used in the decoder. Choose among ['transpose', 'bilinear'].
cfg.model.up_mode = 'transpose'

# Enable (True) or disable (False) batch normalization after every convolutional layer.
cfg.model.do_BN = True

# Activate (True) or deactivate (False) the learnable bias of the convolutional layers. The learnable bias is
# automatically switched off if the convolutional layer is followed by batch normalization.
cfg.model.bias_conv_layer = True


# Specify the number of input views of the 'geom-multiview' setting. Choose among ['3-view', '4-view', '5-view'].
cfg.multiview.config = '3-view'


# Specify how the training data samples are created. Set this parameter to True to combine each training image (pair)
# with every initial DSM patch to form a data sample (i.e., generalization of ResDepth-stereo across viewpoints,
# see Section 3, Generalization in the paper). Set this parameter to False to randomly assign an image (pair) to every
# initial DSM patch to form a data sample (if multiple training images are available).
cfg.stereopair_settings.use_all_stereo_pairs = True

# Specify how the training data samples are created. Set this parameter to True to randomly flip the order of the
# ortho-images in every sample. Set this parameter to False to maintain the order of the ortho-images according to
# the image pair lists.
cfg.stereopair_settings.permute_images_within_pair = True


# GLOBAL SETTING: Number of training samples (DSM patches) per dataset/geographic region (i.e., the same number of
# training samples are randomly sampled from every dataset/geographic region if multiple training datasets are listed
# in the configuration file). Alternatively, this parameter can be specified for each dataset/geographic region
# individually to overwrite this global setting.
cfg.training_settings.n_training_samples = 20000

# Tile size in pixels.
cfg.training_settings.tile_size = 256

# Activate (True) or deactivate (False) data augmentation (random rotation by multiples of 90 degrees and random
# flipping along the horizontal and vertical axes).
cfg.training_settings.augment = True

# Batch size.
cfg.training_settings.batch_size = 20

# Number of training epochs.
cfg.training_settings.n_epochs = 2000

# Loss function. Choose among ['L1'].
cfg.training_settings.loss = 'L1'


# Optimizer used for optimization. Choose among ['Adam', 'SGD'].
cfg.optimizer.name = 'Adam'

# Initial learning rate.
cfg.optimizer.learning_rate = 2e-04

# Weight decay.
cfg.optimizer.weight_decay = 1e-05


# Enable (True) or disable (False) a learning rate scheduler.
cfg.scheduler.enabled = True

# Specify the learning rate scheduler. Choose among ['ReduceLROnPlateau', 'StepLR', 'ExponentialLR'].
cfg.scheduler.name = 'StepLR'

# Specify additional arguments of the learning rate scheduler using the parameter name convention of PyTorch.
cfg.scheduler.settings = edict()
# Example: cfg.scheduler.settings.step_size = 50


# GLOBAL SETTING: Specify if and how every dataset/geographic region is split into geographically separate stripes for
# training, validation, and testing. Choose among ['5-crossval_vertical', '5-crossval_horizontal', 'entire'], where:
# '5-crossval_vertical':    splits the region into five equally large and mutually exclusive vertical (north-south
#                           oriented) stripes
# '5-crossval_horizontal':  splits the region into five equally large and mutually exclusive horizontal (west-east
# #                         oriented) stripes
# 'entire':                 uses the entire raster either for training, validation, or testing
#                           (specified by the parameter 'area_type' of the respective dataset)
# Alternatively, this parameter can be specified for each dataset/geographic region individually to overwrite this
# global setting.
cfg.general.allocation_strategy = '5-crossval_vertical'

# GLOBAL SETTING: Specify which of the five stripes is used as the test stripe. The validation stripe is located to the
# right/bottom (east/south) of the test stripe (cyclic order).
# Alternatively, this parameter can be specified for each dataset/geographic region individually to overwrite this
# global setting.
cfg.general.test_stripe = 0

# Number of workers used in the data loader.
cfg.general.workers = 4

# Set the random seed for reproducibility.
cfg.general.random_seed = 0

# Frequency (in terms of number of training epochs) with which the model parameters are stored to disk.
cfg.general.save_model_rate = 20

# Specify after how many training epochs the validation data is evaluated.
cfg.general.evaluate_rate = 1


# The name of the results directory consists of the code execution day and time and the suffix specified below.
cfg.output.suffix = ''

# Export the model architecture to text file.
cfg.output.plot_model_txt = False
