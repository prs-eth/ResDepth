import numpy as np
import torch
import sys
import itertools

from lib import data_allocation, data_normalization, fdutil, rasterutils, torch_transforms
from lib.arguments import INPUT_CHANNELS
from lib.validate_arguments import is_string, valid_tile_size, is_boolean, is_positive_integer


SAMPLING_STRATEGIES = ['train', 'val', 'test']


class DsmOrthoDataset(torch.utils.data.Dataset):
    """
    Dataset class that generates the input to ResDepth.

    :param dataset:         EasyDict, dictionary consisting of the following keys:
        mandatory keys:
            raster_gt:      str, path to the ground truth DSM GeoTiff raster (optional key if sampling_strategy=='test')
            raster_in:      str, path to the initial DSM GeoTiff raster
            area_defn:      dictionary, defines one or multiple rectangularly-shaped geographic regions from which
                            DSM patches will be sampled. The dictionary is composed of the following key-value pairs:
                            x_extent:   list of n tuples, where n denotes the number of rectangular regions (stripes).
                                        Each tuple defines the upper-left and lower-right x-coordinate of a rectangular
                                        region (stripe).
                            y_extent:   list of n tuples, where n denotes the number of rectangular regions (stripes).
                                        Each tuple defines the upper-left and lower-right y-coordinate of a rectangular
                                        region (stripe).

                            Assumption: The i.th tuple of x_extent and i.th tuple of y_extent define a geographically
                                        rectangular region (stripe).

                            Specify area_defn as follows if the entire raster should be used:
                                area_defn = {'x_extent': [(0, cols - 1)], 'y_extent': [(0, rows - 1)]}
                            where cols denotes the number of columns and rows the number of rows of the DSM raster.

        optional keys:
            image_list:     list of strings, paths of the precomputed ortho-rectified panchromatic satellite images
            image_pairs:    list of tuples of equal length m, where m denotes the number of input image views per sample
                            (e.g., m=1 equals mono guidance, m=2 equals stereo guidance). Each integer specifies an
                            image index w.r.t. the images listed in image_list to define the image (pairs) along with
                            raster_in (initial DSM) as input to ResDepth.
                            Example:  image_pairs = [(1,2), (2,6), (1,6)]
                                      defines three stereo image pairs (1,2), (2,6) and (1,6), where the image
                                      indices refer to the images listed in image_list

                            Note: Define a single tuple (image pair) if sampling_strategy=='test'.

            name:           str, name of the dataset (optional dataset identifier)
            n_samples:      int, number of DSM patches randomly sampled from the geographic region area_defn
                            (mandatory key only if sampling_strategy=='train')

    :param input_channels:          str, input channel configuration (model architecture),
                                    choose among ['geom-multiview', 'geom-stereo', 'geom-mono', 'stereo', 'geom']
    :param tile_size:               int, tile size in pixels
    :param sampling_strategy:       str, choose among ['train', 'val', 'test']
    :param stride:                  int, stride of the validation or test DSM patches in pixels
                                    default:    stride = tile_size/2    if sampling_strategy=='test'
                                                stride = tile_size      if sampling_strategy=='val'
    :param transform_dsm:           bool, True to normalize the DSMs, False otherwise
    :param transform_orthos:        bool, True to normalize the ortho-images, False otherwise
    :param dsm_mean:                float, DSM normalization parameter, mean height
                                    ('None' to center each DSM data sample to its mean height)
    :param dsm_std:                 float, DSM normalization parameter, standard deviation in height
    :param ortho_mean:              float, ortho-image normalization parameter, mean radiance
                                    ('None' to center the ortho-images of each data sample to its mean radiance)
    :param ortho_std:               float, ortho-image normalization parameter, standard deviation in radiance
    :param augment:                 bool, True to activate data augmentation (random rotation by multiples of 90 degrees
                                    as well as random flipping along the horizontal and vertical axes), False otherwise
    :param use_all_stereo_pairs:    bool:
                                    if True:    combines each image (pair) in dataset.image_pairs with the same initial
                                                DSM patch to form a data sample
                                                (number of samples corresponds to n_samples * N, where N denotes the
                                                number of images or image pairs and n_samples the number of DSM patches)
                                    if False:   randomly assigns an image (pair) in dataset.image_pairs to each initial
                                                DSM patch to form a data sample (assignment kept fixed once sampled;
                                                number of samples corresponds to the number of DSM patches)
    :param permute_images_within_pair:  bool:
                                        if True:  randomly flips the order of the ortho-images in every sample
                                        if False: maintains the order of the ortho-images according to the definition in
                                                  dataset.image_pairs
    """

    def __init__(self, dataset, input_channels, tile_size, sampling_strategy, stride=None, transform_dsm=True,
                 transform_orthos=True, dsm_mean=None, dsm_std=1.0, ortho_mean=None, ortho_std=1.0, augment=False,
                 use_all_stereo_pairs=False, permute_images_within_pair=False):

        # Save the input channel configuration
        self.input_channels = input_channels

        # Save the tile size
        self.tile_size = tile_size

        # Save the sampling strategy
        self.sampling_strategy = sampling_strategy

        # Save the stride for creating a regular grid of DSM patches (relevant if sampling_strategy in ['val', 'test'])
        if stride is None and self.sampling_strategy == 'test':
            self.stride = int(self.tile_size * 0.5)
        elif stride is None and self.sampling_strategy == 'val':
            self.stride = self.tile_size
        else:
            self.stride = stride

        # Save the data augmentation mode
        self.augment = augment

        # Save the flag to activate/deactivate data normalization
        self.transform_dsm = transform_dsm
        self.transform_orthos = transform_orthos

        # Save the data normalization parameters.
        # If the mean is set to None, each patch will be individually centered to its mean height or radiance.
        self.dsm_mean = dsm_mean
        self.dsm_std = dsm_std
        self.ortho_mean = ortho_mean
        self.ortho_std = ortho_std

        # Save the handling of the image pairs
        # (use all pairs vs. random sampling of a pair per data sample, random permutation of the images within a pair
        # activated/deactivated)
        self.use_all_stereo_pairs = use_all_stereo_pairs
        self.permute_images_within_pair = permute_images_within_pair

        # Verify the input arguments
        self._verify_inputs(dataset)

        # Save the dataset arguments
        self.raster_in = dataset.raster_in
        self.area_defn = dataset.area_defn

        if 'raster_gt' in dataset:
            self.raster_gt = dataset.raster_gt
        else:
            self.raster_gt = None

        if self.input_channels != 'geom':
            self.image_list = dataset.image_list
            self.image_pairs = dataset.image_pairs

        if 'name' in dataset:
            self.name = dataset.name
        else:
            self.name = None

        if 'n_samples' in dataset:
            self.n_samples = dataset.n_samples
        else:
            self.n_samples = None

        # Load DSM rasters and ortho-rectified satellite images
        self._load_data()

        # Determine/Sample patch positions
        self._determine_patches()

    def __len__(self):
        return self.total_dsm_ortho_samples

    def __getitem__(self, index):

        # Extract the patch position
        y, x = self.patch_position[index]

        # ----------------------------------------- Load DSM sample ----------------------------------------- #
        # Extract the initial DSM patch (assumption: the initial DSM and ground truth DSM are co-registered)
        dsm_input = self.dsm_input[y:y+self.tile_size, x:x+self.tile_size]  # [tile_size, tile_size]

        # Extract the ground truth DSM patch
        if self.raster_gt is not None:
            dsm_target = self.dsm_target[y:y+self.tile_size, x:x+self.tile_size]  # [tile_size, tile_size]
        else:
            dsm_target = np.nan

        # ------------------------------------------ Get loss masks ------------------------------------------ #
        if self.sampling_strategy == 'train':
            loss_mask = self._get_dsm_loss_mask(dsm_target, self.nodata)  # [tile_size, tile_size]
            patch_valid_pixels = np.full((4,), fill_value=np.nan)
        else:
            # Get valid pixels due to overlapping grid tiles
            patch_valid_pixels = self.patch_valid_pixels[index]

            # Get mask of valid pixels
            if self.raster_gt is not None:
                loss_mask = self._get_dsm_loss_mask(dsm_target, self.nodata, patch_valid_pixels)  # [tile_size, tile_size]
            else:
                loss_mask = np.nan

        # ---------------------------------------- DSM normalization ----------------------------------------- #
        if self.transform_dsm:
            if not self.dsm_mean:
                # Compute the mean height of the initial DSM patch
                dsm_mean = np.ma.mean(np.ma.masked_where(dsm_input == self.nodata, dsm_input))
            else:
                # Use user-specified mean
                dsm_mean = self.dsm_mean

            # Perform DSM normalization
            transform = data_normalization.get_transform(dsm_mean, self.dsm_std)
            dsm_input = transform(dsm_input)        # [1, tile_size, tile_size]
            if self.raster_gt is not None:
                dsm_target = transform(dsm_target)  # [1, tile_size, tile_size]
        else:
            # Add third dimension manually (grayscale image)
            # Note: transform() adds the third dimension automatically
            dsm_input = torch.from_numpy(dsm_input).unsqueeze(0)        # [1, tile_size, tile_size]
            dsm_mean = 0
            if self.raster_gt is not None:
                dsm_target = torch.from_numpy(dsm_target).unsqueeze(0)  # [1, tile_size, tile_size]

        # ------------------------------------ Load satellite image views ------------------------------------ #
        if self.input_channels != 'geom':
            # Dimension of orthos is [v, tile_size, tile_size], where v equals the number of image views per data
            # sample (e.g., v=2 for stereo and v=1 for mono)

            # Extract the image (pair) index/indices
            img_pair_index = self.image_pair_indices[index]
            img_pair = self.image_pairs[img_pair_index]

            # Extract the ortho-rectified satellite image patches
            orthos = self.orthos[y:y+self.tile_size, x:x+self.tile_size, img_pair].transpose((2, 0, 1))

            if self.permute_images_within_pair:
                perm = np.arange(orthos.shape[0])
                np.random.shuffle(perm)
                orthos = orthos[perm]

            # Satellite image normalization
            if self.transform_orthos:
                if not self.ortho_mean:
                    # Compute the mean radiance of the image patches
                    ortho_mean = orthos.mean()
                else:
                    # Use user-specified mean
                    ortho_mean = self.ortho_mean

                # Get ortho transformation
                transform = data_normalization.get_transform(ortho_mean, self.ortho_std)
                for j in range(orthos.shape[0]):
                    orthos[j, ...] = transform(orthos[j, ...])

            orthos = torch.from_numpy(orthos)

            if self.input_channels != 'stereo':
                # Concatenate the DSM patch and the satellite image view(s) to form the network input to ResDepth
                # Dimension of inputs: [v+1, tile_size, tile_size],
                # where input[0, :, :] contains the initial DSM patch
                # and   input[1::, :, :] contains the satellite image view(s)
                inputs = torch.cat((dsm_input, orthos), axis=0)
            else:
                inputs = orthos.clone()
        else:
            # Network input
            inputs = dsm_input.clone()

        # Adjust the dimensions of the loss mask to match the dimensions of inputs
        if self.raster_gt is not None:
            loss_mask = loss_mask.repeat(1, 1, 1)  # [1, tile_size, tile_size]

        # ------------------------------------------ Augmentation ------------------------------------------ #
        if self.sampling_strategy == 'train' and self.augment:
            augment_batch = torch_transforms.Compose([
                torch_transforms.Rotate(),
                torch_transforms.RandomVerticalFlip(),
                torch_transforms.RandomHorizontalFlip(),
            ])

            if self.raster_gt is not None:
                augmented = augment_batch(torch.cat((loss_mask, dsm_target, inputs), axis=0))
                loss_mask = augmented[0, ...].unsqueeze(0)
                dsm_target = augmented[1, ...].unsqueeze(0)
                inputs = augmented[2::, ...]

            else:
                inputs = augment_batch(inputs)

        if self.raster_gt is not None:
            loss_mask = loss_mask.type(torch.bool)

        return {'input': inputs,                                    # first initial DSM (0.th channel), then image views
                'target': dsm_target,                               # ground truth DSM
                'patch_offset_x': x, 'patch_offset_y': y,           # upper-left image coordinates of the DSM patch
                'nodata': self.nodata,                              # nodata value of the DSM
                'loss_mask': loss_mask,                             # boolean loss mask
                'dsm_mean': dsm_mean, 'dsm_std': self.dsm_std,      # DSM normalization parameters
                'patch_valid_pixels_uly': patch_valid_pixels[0],
                'patch_valid_pixels_ulx': patch_valid_pixels[1],    # pixels of the DSM patch that do not overlap
                'patch_valid_pixels_lry': patch_valid_pixels[2],    # with any other patch
                'patch_valid_pixels_lrx': patch_valid_pixels[3]
                }

    def _load_data(self):
        # Load the initial and ground truth DSM raster
        self.dsm_input_gdal = rasterutils.load_raster(self.raster_in)
        self.dsm_input = self.dsm_input_gdal.GetRasterBand(1).ReadAsArray().astype(np.float32)

        if self.raster_gt is not None:
            self.dsm_target_gdal = rasterutils.load_raster(self.raster_gt)
            self.dsm_target = self.dsm_target_gdal.GetRasterBand(1).ReadAsArray().astype(np.float32)

            # Save nodata value of the DSM data
            self.nodata = np.array(self.dsm_target_gdal.GetRasterBand(1).GetNoDataValue()).astype(np.float32)
        else:
            self.nodata = np.array(self.dsm_input_gdal.GetRasterBand(1).GetNoDataValue()).astype(np.float32)

        # Load the ortho-rectified images
        if self.input_channels != 'geom':
            extent = rasterutils.get_raster_extent(self.dsm_input_gdal)

            self.orthos = np.zeros((extent['rows'], extent['cols'], len(self.image_list)), dtype=np.float32)
            for j, img in enumerate(self.image_list):
                ds = rasterutils.load_raster(img)
                self.orthos[..., j] = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    def _determine_patches(self):
        if self.sampling_strategy == 'train':
            # Get tuples of valid DSM patch positions (upper-left image coordinates) to sample patches
            valid_positions = data_allocation.indices_from_area_defn(self.area_defn, self.tile_size)

            # Sample training patches: sample upper-left patch coordinates
            indices = np.random.choice(len(valid_positions), self.n_samples, replace=False)
            sampled_positions = [valid_positions[i] for i in indices]

            # Sample ortho-rectified satellite images
            if self.input_channels == 'geom-stereo' and len(self.image_pairs) > 1:
                if self.use_all_stereo_pairs:
                    # Number of image pairs
                    n = len(self.image_pairs)

                    # Repeat the sampled DSM patch positions n times, where n is the number of image pairs
                    # Example:
                    # indices of the sampled patch positions (=indices) are [2, 7, 45] and n=3:
                    # np.repeat([2, 7, 45], 3) => [2, 2, 2, 7, 7, 7, 45, 45, 45]
                    # Interpretation: At each sampled location, the DSM patch will be extracted three (= n) times to be
                    # combined with a stereo image pair
                    indices_repeat = np.repeat(indices, n)
                    self.patch_position = [valid_positions[i] for i in indices_repeat]

                    # Combine each sampled DSM patch with every stereo image pair to form a training sample
                    # Example for n=3:
                    # self.image_pair_indices = [0, 1, 2, 0, 1, 2, 0, 1, 2]
                    self.image_pair_indices = np.tile(np.linspace(0, n, n, endpoint=False, dtype=np.int64),
                                                      self.n_samples)

                    # Save the number of samples
                    self.total_dsm_samples = self.n_samples
                    self.total_dsm_ortho_samples = self.n_samples * n

                else:
                    # Save the sampled DSM patch positions (upper-left image coordinates)
                    self.patch_position = sampled_positions

                    # Sample one random image pair per DSM patch
                    self.image_pair_indices = np.random.choice(len(self.image_pairs), self.n_samples, replace=True)

                    # Save the number of training samples
                    self.total_dsm_samples = self.n_samples
                    self.total_dsm_ortho_samples = self.n_samples

            else:
                # Save the sampled DSM patch positions (upper-left image coordinates)
                self.patch_position = sampled_positions

                # Save the number of training samples
                self.total_dsm_samples = self.n_samples
                self.total_dsm_ortho_samples = self.n_samples

                # Always use the same input image view(s), e.g., the input image (pair) with index 0
                # (dataset consists of one image (pair) only)
                self.image_pair_indices = np.zeros(self.n_samples, dtype=np.int64)

        # Get a regular grid of patches without overlap
        elif self.sampling_strategy == 'val':
            # Get validation patches: create a regular grid of non-overlapping DSM patches
            positions, patch_valid_pixels = \
                rasterutils.create_regular_grid(self.area_defn, tile_size=self.tile_size, stride=self.stride)

            # Evaluate each image pair
            if self.input_channels != 'geom':
                # Number of image pairs
                n = len(self.image_pairs)

                # Repeat the DSM patch positions n times, where n is the number of image pairs
                self.patch_position = positions * n

                # Save the valid pixels to be evaluated (non-valid pixels: overlapping patches at the region boundary)
                self.patch_valid_pixels = patch_valid_pixels * n

                # Combine each DSM patch with every image pair
                self.image_pair_indices = np.repeat(np.linspace(0, n, n, endpoint=False, dtype=np.int64),
                                                    len(positions))

                # Save the number of validation samples
                self.total_dsm_samples = len(positions)
                self.total_dsm_ortho_samples = len(positions) * n

            else:
                # Save the number of validation samples
                self.total_dsm_samples = len(positions)
                self.total_dsm_ortho_samples = len(positions)

                # Save the DSM patch positions (upper-left image coordinates)
                self.patch_position = positions

                # Save the valid pixels to be evaluated (non-valid pixels: overlapping patches at the raster boundary)
                self.patch_valid_pixels = patch_valid_pixels

                # Always use the same input image view(s), e.g., the input image (pair) with index 0
                self.image_pair_indices = np.zeros(len(positions), dtype=np.int64)

        # Get a regular grid of patches with overlap
        else:
            # i.e., self.sampling_strategy == 'test'

            # Create a regular grid of overlapping DSM patches
            positions, patch_valid_pixels = \
                rasterutils.create_regular_grid(self.area_defn, tile_size=self.tile_size, stride=self.stride)

            # Save the number of test samples
            self.total_dsm_samples = len(positions)
            self.total_dsm_ortho_samples = len(positions)

            # Save the DSM patch positions (upper-left image coordinates)
            self.patch_position = positions

            # Save the valid pixels to be evaluated (used to perform linear blending of the tiles)
            self.patch_valid_pixels = patch_valid_pixels

            # Always use the same input image view(s), e.g., the input image (pair) with index 0
            self.image_pair_indices = np.zeros(len(positions), dtype=np.int64)

    @staticmethod
    def _get_dsm_loss_mask(dsm, nodata, patch_valid_pixels=None):
        """
        Returns a boolean mask, where a pixels equals to:
            True  if the pixel has a valid value and contributes to the loss (i.e., the pixel is within a
                  non-overlapping patch region and does not contain the nodata value).
            False otherwise.
        :param dsm:                 torch array of dimension [1, tile_size, tile_size], ground truth DSM patch
        :param nodata:              single element float numpy array, nodata value of dsm
        :param patch_valid_pixels:  tuple (uly, ulx, lry, lrx), defines the pixels of dsm that do not overlap with any
                                    other DSM patch (i.e., these pixels will be taken into account when evaluating
                                    the loss)
        :return:                    torch tensor, integer mask
        """

        # mask1: a pixel is True if it should be evaluated in the loss
        # (i.e., true for non-overlapping pixels of a regular grid of raster patches)
        valid = np.zeros(dsm.shape)

        if patch_valid_pixels is not None:
            uly, ulx, lry, lrx = patch_valid_pixels
            valid[..., uly:lry+1, ulx:lrx+1] = dsm[..., uly:lry+1, ulx:lrx+1]
        else:
            if isinstance(dsm, torch.Tensor):
                valid = dsm.clone()
            else:
                valid = np.copy(dsm)

        # True if the respective ground truth pixels are not masked out (due to overlapping patches)
        mask1 = valid != 0

        # mask2: a pixel is True if it contains a valid height
        mask2 = dsm != nodata

        # Combine the two masks
        mask_final = np.logical_and(mask1, mask2)

        return torch.from_numpy(mask_final).type(torch.bool)

    def _verify_inputs(self, dataset):
        if self.input_channels not in INPUT_CHANNELS:
            raise ValueError(f"ERROR: Unknown input channel configuration: '{self.input_channels}'.\nChoose among "
                             f"{INPUT_CHANNELS} to specify 'input_channels'.\n")

        if not valid_tile_size(self.tile_size, 'tile_size'):
            sys.exit(1)

        if self.sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"ERROR: Unknown sampling strategy: '{self.sampling_strategy}'.\nChoose among "
                             f"{SAMPLING_STRATEGIES} to specify 'sampling_strategy'.\n")

        if self.stride is not None and not is_positive_integer(self.stride, 'stride'):
            raise ValueError(f"ERROR: Invalid argument 'stride: '{self.stride}'. Specify a positive integer'.\n")

        if not is_boolean(self.transform_dsm, 'transform_dsm'):
            sys.exit(1)

        if not is_boolean(self.transform_orthos, 'transform_orthos'):
            sys.exit(1)

        if self.dsm_mean is not None and not isinstance(self.dsm_mean, float):
            raise ValueError("ERROR: Invalid argument 'dsm_mean'. Specify a float.\n")

        if not isinstance(self.dsm_std, float) or self.dsm_std < 0:
            raise ValueError("ERROR: Invalid argument 'dsm_std'. Specify a positive float.\n")

        if self.ortho_mean is not None and not isinstance(self.ortho_mean, float):
            raise ValueError("ERROR: Invalid argument 'ortho_mean'. Specify a float.\n")

        if self.ortho_std is not None:
            if not isinstance(self.ortho_std, float) or self.ortho_std < 0:
                raise ValueError("ERROR: Invalid argument 'ortho_std'. Specify a positive float.\n")

        if not is_boolean(self.augment, 'augment'):
            sys.exit(1)

        if not is_boolean(self.use_all_stereo_pairs, 'use_all_stereo_pairs'):
            sys.exit(1)

        if not is_boolean(self.permute_images_within_pair, 'permute_images_within_pair'):
            sys.exit(1)

        # Verify 'dataset' argument
        if not isinstance(dataset, dict):
            raise ValueError("ERROR: Invalid 'dataset' argument. Enter a dictionary.\n")

        # Verify that the initial DSM exists
        if 'raster_in' not in dataset:
            print("ERROR: Missing argument 'raster_in' in 'dataset'. Specify the path of the input depth/height "
                  "raster.\n")
            sys.exit(1)
        elif not is_string(dataset.raster_in, 'raster_in'):
            sys.exit(1)
        elif not fdutil.file_exists(dataset.raster_in):
            print('ERROR: Input depth/height raster does not exist:\n{}\n'.format(dataset.raster_in))
            sys.exit(1)

        # Verify that the ground truth DSM exists
        if self.sampling_strategy in ['train', 'val'] and 'raster_gt' not in dataset:
            print("ERROR: Missing argument 'raster_gt' in 'dataset'. Specify the path of the ground truth depth/height "
                  "raster.\n")
            sys.exit(1)

        ds_in = rasterutils.load_raster(dataset.raster_in)
        geotransform_in = ds_in.GetGeoTransform()

        if 'raster_gt' in dataset:
            if not is_string(dataset.raster_gt, 'raster_gt'):
                sys.exit(1)
            elif not fdutil.file_exists(dataset.raster_gt):
                print(f"ERROR: Ground truth depth/height raster does not exist:\n{dataset.raster_gt}\n")
                sys.exit(1)

            ds_target = rasterutils.load_raster(dataset.raster_gt)
            geotransform_target = ds_target.GetGeoTransform()

            # Verify that the initial and ground truth DSM have the same spatial dimension (width and height)
            if ds_in.RasterXSize != ds_target.RasterXSize or ds_in.RasterYSize != ds_target.RasterYSize:
                raise ValueError('ERROR: Initial DSM and ground truth DSM have different spatial dimensions.\n')

            # Verify that the initial and ground truth DSM have the same spatial resolution
            if geotransform_target[1] != geotransform_in[1] or geotransform_target[5] != geotransform_in[5]:
                raise ValueError('ERROR: Initial DSM and ground truth DSM have a different spatial resolutions.\n')
            ds_target = None
        ds_in = None

        # Save the resolution of each raster (sanity check: each image/DSM raster needs to exhibit the same
        # spatial resolution)
        resolution_dsm = geotransform_in[1]
        resolutions_images = []

        if self.input_channels in ['geom-multiview', 'geom-stereo', 'geom-mono', 'stereo']:
            # Verify that the paths to the precomputed ortho-rectified images exist
            if 'image_list' not in dataset:
                print("ERROR: Missing argument 'image_list' in 'dataset'. Specify a list of paths to the precomputed "
                      "ortho-rectified images.\n")
                sys.exit(1)
            elif not isinstance(dataset.image_list, list):
                raise ValueError("ERROR: Invalid 'image_list' in 'dataset'. Enter a list of strings.\n")
            else:
                for img in dataset.image_list:
                    if not fdutil.file_exists(img):
                        print('ERROR: Cannot find the precomputed ortho-rectified image: {}\n'.format(img))
                        sys.exit(1)
                    else:
                        ds = rasterutils.load_raster(img)
                        resolutions_images.append(ds.GetGeoTransform()[1])
                        ds = None

                # Verify that all images have the same spatial resolution
                if len(set(resolutions_images)) > 1:
                    print("ERROR: Ortho-rectified images exhibit different spatial resolutions.\n")
                    sys.exit(1)

                # Verify that the DSMs and ortho-rectified images have the same spatial resolution:
                if resolution_dsm != resolutions_images[0]:
                    print('ERROR: The DSMs and ortho-rectified images do not have the same spatial resolution.\n')
                    sys.exit(1)

            # Verify the list of image pairs
            if 'image_pairs' not in dataset:
                print("ERROR: Missing argument 'image_pairs' in 'dataset'. Specify a list of tuples to define "
                      "the image pairs, where the indices refer to the images listed in 'image_list'.\n")
                sys.exit(1)
            elif not isinstance(dataset.image_pairs, list):
                raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. Enter a list of tuples.\n")
            else:
                if self.sampling_strategy == 'test':
                    if len(dataset.image_pairs) > 1:
                        raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. "
                                         "Specify a single image pair.\n")

                n_images_per_pair = []
                for pair in dataset.image_pairs:
                    n_images_per_pair.append(len(pair))
                    if not isinstance(pair, tuple):
                        raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. Enter a list of tuples.\n")
                if len(set(n_images_per_pair)) > 1:
                    raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. "
                                     "Enter a list of tuples, where each tuple has the same length.\n")

                # Find maximum image index
                max_index = np.asarray(list(itertools.chain(*dataset.image_pairs))).max()
                if max_index > len(dataset.image_list):
                    raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. "
                                     "Largest image index exceeds the number of images given in 'image_list'.\n")

        if 'area_defn' not in dataset or 'x_extent' not in dataset.area_defn or \
                'y_extent' not in dataset.area_defn or not \
                isinstance(dataset.area_defn.x_extent, list) or not isinstance(dataset.area_defn.y_extent, list):
            print("ERROR: Missing argument 'area_defn' in 'dataset'. Specify a dictionary with the keys 'x_extent' "
                  "and 'y_extent'.\n")
            sys.exit(1)
        else:
            extent = rasterutils.get_raster_extent(dataset.raster_in)
            x_extent = dataset.area_defn.x_extent
            y_extent = dataset.area_defn.y_extent

            if not isinstance(x_extent, list) or not isinstance(y_extent, list):
                raise ValueError("ERROR: Invalid argument 'x_extent' or 'y_extent' in 'dataset'. "
                                 "Enter a list of tuples.\n")

            for area in x_extent:
                if area[0] < 0 or area[1] >= extent['cols']:
                    raise ValueError("ERROR: Invalid argument 'x_extent' in 'dataset'. Specify minimum and maximum "
                                     f"pixel coordinates in [0, {extent['cols'] - 1}].\n.")

            for area in y_extent:
                if area[0] < 0 or area[1] >= extent['rows']:
                    raise ValueError("ERROR: Invalid argument 'y_extent' in 'dataset'. Specify minimum and maximum "
                                     f"pixel coordinates in [0, {extent['rows'] - 1}].\n.")

        if self.sampling_strategy == 'train':
            if 'n_samples' not in dataset:
                print("ERROR: Missing argument 'n_samples' in 'dataset'. Specify the number of training samples.\n")
                sys.exit(1)
            elif not isinstance(dataset.n_samples, int) or dataset.n_samples <= 0:
                raise ValueError("ERROR: Invalid argument 'n_samples' in 'dataset'. Specify a positive integer.\n")

        # Verify the number of images per image pair
        if self.input_channels in ['stereo', 'geom-stereo'] and len(dataset.image_pairs[0]) != 2:
            raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. Enter a list consisting of one or"
                             "multiple tuples with length 2 (image pair(s) composed of 2 images).\n")

        elif self.input_channels == 'geom-mono' and len(dataset.image_pairs[0]) != 1:
            raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. Enter a list consisting of a single"
                             "tuple with length 1 (single image).\n")

        elif self.input_channels == 'geom-multiview' and len(dataset.image_pairs[0]) < 2:
            raise ValueError("ERROR: Invalid argument 'image_pairs' in 'dataset'. Enter a list consisting of a single"
                             "tuple with length n, where n>2 (n-view).\n")

        # Save GSD
        self.GSD = resolution_dsm

        # Convert normalization parameters to np.float32
        if self.dsm_mean is not None:
            self.dsm_mean = np.asarray(self.dsm_mean).astype(np.float32)
        self.dsm_std = np.asarray(self.dsm_std).astype(np.float32)
        if self.ortho_mean is not None:
            self.ortho_mean = np.asarray(self.ortho_mean).astype(np.float32)
        self.ortho_std = np.asarray(self.ortho_std).astype(np.float32)
