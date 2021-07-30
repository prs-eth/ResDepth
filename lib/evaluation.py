from easydict import EasyDict as edict
import logging
import logging.config
import numpy as np
from osgeo import gdal
import torch

from lib import data_normalization, fdutil, rasterutils, utils


def compute_residuals(raster, raster_gt, nodata, mask_gt=None):
    """
    Computes the residual errors. A positive error means that the predicted height is larger than the reference value
    and conversely for a negative error.

    :param raster:      np.array, DSM to be evaluated
    :param raster_gt:   np.array, reference (ground truth) DSM
    :param nodata:      float, nodata value of the DSMs
    :param mask_gt:     np.array (boolean), ground truth mask (True indicates valid ground truth heights)
    :return:            np.ma.array, DSM residual errors
    """

    # Mask out invalid ground truth pixels
    if mask_gt is not None:
        mask = np.ma.mask_or(raster_gt == nodata, ~mask_gt)
        raster_gt_masked = np.ma.masked_array(raster_gt, mask=mask)
    else:
        raster_gt_masked = np.ma.masked_where(raster_gt == nodata, raster_gt)

    # Mask out invalid pixels in the input raster
    raster_masked = np.ma.masked_where(raster == nodata, raster)

    # Compute residual errors: raster - raster_gt
    residuals_masked = raster_masked - raster_gt_masked

    return residuals_masked


def truncate_residuals(residuals, threshold):
    """
    Truncates residual errors outside the interval [-threshold, threshold].

    :param residuals:   np.ma.array, DSM residual errors
    :param threshold:   positive float, threshold to truncate the residual errors prior to evaluation
    :return:            np.ma.array, truncated DSM residual errors
    """

    return np.ma.masked_outside(residuals, -threshold, threshold)


def get_statistics(residuals_masked, residual_threshold=None):
    """
    Computes several evaluation metrics using non-truncated residuals and optionally using truncated residuals.

    :param residuals_masked:    np.ma.array, DSM residual errors
    :param residual_threshold:  positive float, threshold to truncate the residual errors prior to evaluation
                                (None to deactivate thresholding)
    :return:                    EasyDict, dictionary with the following key-value pairs
                                (statistics of the non-truncated residual errors):
                                truncation:         boolean, True if the evaluation metrics are additionally computed
                                                    for truncated residual errors, False otherwise
                                count_total:        float, number of valid pixels
                                diff_max:           float, maximum residual error [m]
                                diff_min:           float, minimum residual error [m]
                                MAE:                float, mean absolute error (MAE) [m]
                                RMSE:               float, root mean square error (RMSE) [m]
                                absolute_median:    float, median absolute error (MedAE) [m]
                                median:             float, median error [m]
                                NMAD:               float, normalized median absolute deviation (NMAD) [m]

                                If residual_threshold is not equal to None:
                                truncated:      EasyDict, dictionary with the following key-value pairs
                                                (statistics of the truncated residual errors):
                                                count_total:        float, number of valid pixels
                                                threshold:          =residual_threshold
                                                MAE:                float, mean absolute error (MAE) [m]
                                                RMSE:               float, root mean square error (RMSE) [m]
                                                absolute_median:    float, median absolute error (MedAE) [m]
                                                median:             float, median error [m]
                                                NMAD:               float, normalized median absolute deviation (NMAD) [m]
    """

    stats = edict()
    stats.truncation = True if residual_threshold else False

    if residual_threshold:
        stats.truncated = edict()

        # Compute absolute truncated residual errors
        residuals_truncated = truncate_residuals(residuals_masked, residual_threshold)
        abs_residuals_truncated = np.ma.abs(residuals_truncated)

        # Number of unmasked pixels (= number of valid pixels)
        stats.truncated.count_total = float(np.ma.count(residuals_truncated))
        stats.truncated.threshold = residual_threshold

    # Number of unmasked pixels (= number of valid pixels)
    stats.count_total = float(np.ma.count(residuals_masked))

    # Compute absolute residual errors
    abs_residuals = np.ma.abs(residuals_masked)

    # Minimum and maximum residual error
    stats.diff_max = np.ma.MaskedArray.max(residuals_masked)
    stats.diff_min = np.ma.MaskedArray.min(residuals_masked)

    # Mean absolute error (MAE)
    stats.MAE = np.ma.mean(abs_residuals)

    # Root mean square error (RMSE)
    stats.RMSE = np.ma.sqrt(np.ma.mean(abs_residuals ** 2))

    # Median absolute error
    stats.absolute_median = np.ma.median(abs_residuals)

    # Median error
    stats.median = np.ma.median(residuals_masked)

    # Normalized median absolute deviation (NMAD)
    abs_diff_from_med = np.ma.abs(residuals_masked - stats.absolute_median)
    stats.NMAD = 1.4826 * np.ma.median(abs_diff_from_med)

    if stats.truncation:
        stats.truncated.MAE = np.ma.mean(abs_residuals_truncated)
        stats.truncated.RMSE = np.ma.sqrt(np.ma.mean(abs_residuals_truncated ** 2))
        stats.truncated.absolute_median = np.ma.median(abs_residuals_truncated)
        stats.truncated.median = np.ma.median(residuals_truncated)
        abs_diff_from_med = np.ma.abs(residuals_truncated - stats.truncated.absolute_median)
        stats.truncated.NMAD = 1.4826 * np.ma.median(abs_diff_from_med)

    return stats


def print_statistics(stats, logger, print_min_max=True):
    """
    Prints the evaluation metrics computed by the function get_statistics().

    :param stats:           EasyDict, dictionary returned by the function get_statistics()
    :param logger:          logger instance
    :param print_min_max:   boolean, True to print minimum and maximum residual errors, False otherwise
    """

    if print_min_max:
        logger.info('Maximum residual error [m]:\t\t\t\t\t\t{:10.3f} m'.format(stats.diff_max))
        logger.info('Minimum residual error [m]:\t\t\t\t\t\t{:10.3f} m'.format(stats.diff_min))

    # Evaluation metrics: non-truncated residual errors
    logger.info('Mean absolute residual error (MAE) [m]:\t\t\t\t\t{:10.3f} m'.format(stats.MAE))
    logger.info('RMSE residual error [m]:\t\t\t\t\t\t{:10.3f} m'.format(stats.RMSE))
    logger.info('Absolute median residual error [m]:\t\t\t\t\t{:10.3f} m'.format(stats.absolute_median))
    logger.info('Median residual error [m]:\t\t\t\t\t\t{:10.3f} m'.format(stats.median))
    logger.info('Normalized median absolute deviation (NMAD) [m]:\t\t\t{:10.3f} m\n'.format(stats.NMAD))

    # Evaluation metrics: truncated residual errors
    if stats.truncation:
        logger.info('Truncated mean absolute residual error (MAE) [m]:\t\t\t{:10.3f} m'.format(stats.truncated.MAE))
        logger.info('Truncated RMSE residual error [m]:\t\t\t\t\t{:10.3f} m'.format(stats.truncated.RMSE))
        logger.info('Truncated absolute median residual error [m]:\t\t\t\t{:10.3f} m'.format(stats.truncated.absolute_median))
        logger.info('Truncated median residual error [m]:\t\t\t\t\t{:10.3f} m'.format(stats.truncated.median))
        logger.info('Truncated normalized median absolute deviation (NMAD) [m]:\t\t{:10.3f} m\n'.format(stats.truncated.NMAD))


def evaluate_performance(raster_prediction, ds_raster_input, ds_raster_gt, logger_root, area_defn=None,
                         path_gt_mask=None, path_building_mask=None, path_water_mask=None, path_forest_mask=None,
                         logger_stats=None, residual_threshold=None):
    """
    Computes the evaluation metrics for both the initial DSM and the refined DSM. The error metrics are computed over
    a) all pixels,
    b) building pixels (if path_building_mask is provided),
    c) terrain pixels (if path_building_mask is provided),
    d) terrain pixels excluding water pixels (if path_building_mask and path_water_mask are provided),
    e) terrain pixels excluding water and forest pixels (if path_building_mask, path_water_mask, and path_forest_mask
       are provided).

    Note that the building mask is dilated by two pixels to avoid aliasing at vertical walls. Additionally, the error
    metrics are optionally computed a second time, where the residual errors exceeding residual_threshold are ignored.

    :param raster_prediction:       np.array, refined DSM
    :param ds_raster_input:         gdal.Dataset or str, initial DSM loaded as gdal.Dataset
                                    (or alternatively, path to the initial DSM GeoTiff raster)
    :param ds_raster_gt:            gdal.Dataset or str, ground truth DSM loaded as gdal.Dataset
                                    (or alternatively, path to the ground truth DSM GeoTiff raster)
    :param logger_root:             logger instance (root logger)
    :param area_defn:               dictionary, defines one or multiple rectangularly-shaped geographic regions
                                    for which the performance will be evaluated. The dictionary is composed of the
                                    following key-value pairs:
                                    x_extent:   list of n tuples, where n denotes the number of rectangular regions
                                                (stripes). Each tuple defines the upper-left and lower-right
                                                x-coordinate of a rectangular region (stripe).
                                    y_extent:   list of n tuples, where n denotes the number of rectangular regions
                                                (stripes). Each tuple defines the upper-left and lower-right
                                                y-coordinate of a rectangular region (stripe).

                                    Assumption: The i.th tuple of x_extent and i.th tuple of y_extent define a
                                                geographically rectangular region (stripe).

                                    Specify area_defn as follows if the entire refined DSM should be evaluated
                                    (alternatively, specify area_defn=None):
                                        area_defn = {'x_extent': [(0, cols - 1)], 'y_extent': [(0, rows - 1)]}
                                    where cols denotes the number of columns and rows the number of rows of the raster.
    :param path_gt_mask:            path to the ground truth mask (a pixel value of 1 indicates a valid pixel, whereas
                                    a pixel value of 0 indicates an invalid ground truth height)
    :param path_building_mask:      path to the building mask (a pixel value of 1 indicates a building pixel, whereas
                                    a pixel value of 0 indicates a terrain pixel)
    :param path_water_mask:         path to the water mask (a pixel value of 1 indicates a water pixel, whereas
                                    a pixel value of 0 indicates a non-water pixel)
    :param path_forest_mask:        path to the forest mask (a pixel value of 1 indicates a forest pixel, whereas
                                    a pixel value of 0 indicates a non-forest pixel)
    :param logger_stats:            logger instance to print the statistics (if None, output is print to console)
    :param residual_threshold:      positive float, threshold to truncate the residual errors prior to evaluation
    :return:                        EasyDict, dictionary storing the residual errors of the refined DSM; the dictionary
                                    consists of the following key-value pairs:
                                    all:                        np.array, residual errors evaluated over all pixels
                                    building:                   np.array, residual errors evaluated over building
                                                                pixels only
                                    terrain:                    np.array, residual errors evaluated over terrain pixels
                                                                only
                                    terrain_nowater:            np.array, residual errors evaluated over terrain pixels
                                                                excluding water bodies
                                    terrain_nowater_noforest:   np.array, residual errors evaluated over terrain pixels
                                                                excluding water bodies and forested areas
    """

    if logger_stats is None:
        logger_stats = utils.setup_logger('stats_logger', level=logging.INFO, log_to_console=True, log_file=None)

    data = edict()
    mask = edict()

    # Load the refined DSM
    if isinstance(raster_prediction, gdal.Dataset):
        data.prediction = raster_prediction.GetRasterBand(1).ReadAsArray().astype(np.float64)
    elif isinstance(raster_prediction, np.ndarray):
        data.prediction = raster_prediction.copy().astype(np.float64)
    else:
        logger_root.info('\tLoad the refined DSM...')
        ds = rasterutils.load_raster(raster_prediction)
        data.prediction = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        ds = None

    # Load the ground truth DSM
    if isinstance(ds_raster_gt, gdal.Dataset):
        data.ground_truth = ds_raster_gt.GetRasterBand(1).ReadAsArray().astype(np.float64)
        data.nodata = np.array(ds_raster_gt.GetRasterBand(1).GetNoDataValue()).astype(np.float64)
    else:
        logger_root.info('\tLoad the ground truth DSM...')
        ds = rasterutils.load_raster(ds_raster_gt)
        data.ground_truth = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        data.nodata = np.array(ds.GetRasterBand(1).GetNoDataValue()).astype(np.float64)
        ds = None

    # Load the initial DSM
    if isinstance(ds_raster_input, gdal.Dataset):
        data.initial = ds_raster_input.GetRasterBand(1).ReadAsArray().astype(np.float64)

        # Get GSD [m]
        gsd = ds_raster_input.GetGeoTransform()[1]
    else:
        logger_root.info('\tLoad the initial DSM...')
        ds = rasterutils.load_raster(ds_raster_input)
        data.initial = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)

        # Get GSD [m]
        gsd = ds.GetGeoTransform()[1]
        ds = None

    # Load the ground truth mask to invalidate unreliable ground truth pixels
    if path_gt_mask:
        if not fdutil.file_exists(path_gt_mask):
            logger_root.info('Cannot find the ground truth mask: {}'.format(path_gt_mask))
            logger_root.info('Evaluating the performance by using all ground truth DSM pixels with a valid height.')
        else:
            logger_root.info('\tLoad the ground truth mask...')
            mask.ground_truth, _ = rasterutils.load_mask_raster(path_gt_mask)
    else:
        mask.ground_truth = np.ones(data.ground_truth.shape, dtype=bool)

    # Load the building mask
    if path_building_mask:
        if not fdutil.file_exists(path_building_mask):
            logger_root.info('Cannot find the building mask: {}'.format(path_building_mask))
            logger_root.info('Evaluating the performance over all pixels.')
        else:
            logger_root.info('\tLoad the building mask...')
            # Load the building mask and convert it to a binary mask
            mask_building, mask_nodata = rasterutils.load_mask_raster(path_building_mask)

            # Dilate the building mask by 2 pixels
            mask.building = rasterutils.dilate_mask(mask_building, iterations=2)
            del mask_building

            # Infer the terrain mask by inverting the building mask
            mask.terrain = np.invert(mask.building)
            mask.terrain[mask_nodata] = np.ma.masked
            del mask_nodata

        # Load the water mask
        if path_water_mask:
            if not fdutil.file_exists(path_water_mask):
                logger_root.info('Cannot find the water mask: {}'.format(path_water_mask))
                logger_root.info('Evaluating the performance without excluding water pixels.')
            else:
                logger_root.info('\tLoad the water mask...')
                mask.water, _ = rasterutils.load_mask_raster(path_water_mask)
                mask.terrain_nowater = mask.terrain.copy()
                mask.terrain_nowater[mask.water] = np.ma.masked

        # Load the forest mask
        if path_forest_mask:
            if not fdutil.file_exists(path_forest_mask):
                logger_root.info('Cannot find the forest mask: {}'.format(path_forest_mask))
                logger_root.info('Evaluating the performance without excluding forest pixels.')
            else:
                logger_root.info('\tLoad the forest mask...')
                mask.forest, _ = rasterutils.load_mask_raster(path_forest_mask)

                if 'water' in mask:
                    mask.terrain_nowater_noforest = mask.terrain_nowater.copy()
                    mask.terrain_nowater_noforest[mask.forest] = np.ma.masked
                else:
                    mask.terrain_nowater_noforest = mask.terrain.copy()
                    mask.terrain_nowater_noforest[mask.forest] = np.ma.masked

    if area_defn is not None:
        # Create an area mask (True if a pixel is within the defined region, False otherwise)
        mask.area = np.zeros(data.ground_truth.shape, dtype=bool)

        # Number of regions specified in area_defn
        num_regions = len(area_defn['x_extent'])

        for i in range(num_regions):
            x = area_defn['x_extent'][i]
            y = area_defn['y_extent'][i]
            mask.area[y[0]:y[1]+1, x[0]:x[1]+1] = True

        # Apply the area mask to all other masks
        for key in mask.keys():
            mask[key] = np.logical_and(mask[key], mask.area)

        # Invalidate ground truth pixels outside the evaluation area
        data.ground_truth[~mask.area] = data.nodata

    # Initialize the residual errors before and after the refinement
    residuals = edict({'before': edict(), 'after': edict()})
    stats = edict({'before': edict(), 'after': edict()})

    # Compute overall residual errors before and after the refinement
    logger_root.info('\tCompute overall residual errors before and after the refinement...')
    residuals.before.all = compute_residuals(data.initial, data.ground_truth, data.nodata, mask.ground_truth)
    stats.before.all = get_statistics(residuals.before.all, residual_threshold)
    residuals.after.all = compute_residuals(data.prediction, data.ground_truth, data.nodata, mask.ground_truth)
    stats.after.all = get_statistics(residuals.after.all, residual_threshold)

    # Compute the error metrics for building/non-building pixels only
    if 'building' in mask:
        # Building residuals
        logger_root.info('\tCompute building residual errors before and after the refinement...')
        residuals.after.building = np.ma.masked_array(residuals.after.all, mask=~mask.building)
        stats.before.building = get_statistics(np.ma.masked_array(residuals.before.all, mask=~mask.building),
                                               residual_threshold)
        stats.after.building = get_statistics(residuals.after.building, residual_threshold)

        # Terrain residuals
        logger_root.info('\tCompute terrain residual errors before and after the refinement...')
        residuals.after.terrain = np.ma.masked_array(residuals.after.all, mask=~mask.terrain)
        stats.before.terrain = get_statistics(np.ma.masked_array(residuals.before.all, mask=~mask.terrain),
                                              residual_threshold)
        stats.after.terrain = get_statistics(residuals.after.terrain, residual_threshold)

        if 'water' in mask:
            # Terrain residuals excluding water pixels
            logger_root.info('\tCompute terrain residual errors before and after the refinement '
                             '(excluding water pixels)...')
            residuals.after.terrain_nowater = np.ma.masked_array(residuals.after.all, mask=~mask.terrain_nowater)
            stats.before.terrain_nowater = get_statistics(np.ma.masked_array(residuals.before.all,
                                                                             mask=~mask.terrain_nowater),
                                                          residual_threshold)
            stats.after.terrain_nowater = get_statistics(residuals.after.terrain_nowater, residual_threshold)

        if 'forest' in mask:
            # Terrain residuals excluding water and densely forested pixels
            if 'water' in mask:
                logger_root.info('\tCompute terrain residual errors before and after the refinement '
                                 '(excluding water and densely forested pixels)...')
            else:
                logger_root.info('\tCompute terrain residual errors before and after the refinement '
                                 '(excluding densely forested pixels)...')
            residuals.after.terrain_nowater_noforest = np.ma.masked_array(residuals.after.all,
                                                                          mask=~mask.terrain_nowater_noforest)
            stats.before.terrain_nowater_noforest = get_statistics(np.ma.masked_array(residuals.before.all,
                                                                                      mask=~mask.terrain_nowater_noforest),
                                                                   residual_threshold)
            stats.after.terrain_nowater_noforest = get_statistics(residuals.after.terrain_nowater_noforest,
                                                                  residual_threshold)

    # Compute the size [km^2] of evaluation area
    area_size = float(stats.before.all['count_total'] * gsd * gsd) / 1000000

    # Write statistics
    logger_stats.info('\n\nPerformance Evaluation\n----------------------\n')

    logger_stats.info('Number of pixels:\t\t\t{}'.format(int(stats.before.all['count_total'])))
    logger_stats.info('Area [km^2]:\t\t\t\t{:.2f}\n'.format(area_size))
    if residual_threshold:
        logger_stats.info('Truncation threshold:\t\t\t{:.2f} m\n'.format(residual_threshold))

    logger_stats.info('\nSTATISTICS, OVERALL: INITIAL DSM\n---------------------------------\n')
    # noinspection PyTypeChecker
    print_statistics(stats.before.all, logger_stats)

    logger_stats.info('\nSTATISTICS, OVERALL: REFINED DSM\n--------------------------------\n')
    print_statistics(stats.after.all, logger_stats)

    if 'building' in mask:
        logger_stats.info('\nSTATISTICS, BUILDING PIXELS: INITIAL DSM\n----------------------------------------\n')
        print_statistics(stats.before.building, logger_stats)

        logger_stats.info('\nSTATISTICS, BUILDING PIXELS: REFINED DSM\n----------------------------------------\n')
        print_statistics(stats.after.building, logger_stats)

        logger_stats.info('\nSTATISTICS, TERRAIN PIXELS: INITIAL DSM\n---------------------------------------\n')
        print_statistics(stats.before.terrain, logger_stats)

        logger_stats.info('\nSTATISTICS, TERRAIN PIXELS: REFINED DSM\n---------------------------------------\n')
        print_statistics(stats.after.terrain, logger_stats)

        if 'water' in mask:
            logger_stats.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT WATER: INITIAL DSM\n'
                              '-----------------------------------------------------\n')
            print_statistics(stats.before.terrain_nowater, logger_stats)

            # Write statistics of the refined DSM
            logger_stats.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT WATER: REFINED DSM\n'
                              '-----------------------------------------------------\n')
            print_statistics(stats.after.terrain_nowater, logger_stats)

            if 'forest' in mask:
                logger_stats.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT WATER/FOREST: INITIAL DSM\n'
                                  '------------------------------------------------------------\n')
                print_statistics(stats.before.terrain_nowater_noforest, logger_stats)

                # Write statistics of the refined DSM
                logger_stats.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT WATER/FOREST: REFINED DSM\n'
                                  '------------------------------------------------------------\n')
                print_statistics(stats.after.terrain_nowater_noforest, logger_stats)

        elif 'forest' in mask:
            logger_stats.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT FOREST: INITIAL DSM\n'
                              '------------------------------------------------------\n')
            print_statistics(stats.before.terrain_nowater_noforest, logger_stats)

            # Write statistics of the refined DSM
            logger_stats.info('\nSTATISTICS, TERRAIN PIXELS WITHOUT FOREST: REFINED DSM\n'
                              '------------------------------------------------------\n')
            print_statistics(stats.after.terrain_nowater_noforest, logger_stats)

    return residuals.after


def predict_linear_blend(dataloader, model):
    """
    Applies the model to the initial DSM patches stored in the dataloader and returns the refined DSM patches.
    The refined DSM patches are then merged by linearly blending the refined heights in overlapping patch regions.

    :param dataloader:  torch.utils.data.DataLoader instance, test data
    :param model:       nn.Module instance, the model used for training
    :return:            np.array, refined DSM
                        (same spatial extent as the initial DSM stored in dataloader.dataset.dsm_input)
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Initialize the refined DSM raster
    cols = dataloader.dataset.dsm_input_gdal.RasterXSize
    rows = dataloader.dataset.dsm_input_gdal.RasterYSize
    raster_out = np.zeros((rows, cols))

    tile_size = dataloader.dataset.tile_size
    stride = dataloader.dataset.stride

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            # Extract the offset of the samples w.r.t. the entire raster
            offsets_x = batch['patch_offset_x'].numpy()
            offsets_y = batch['patch_offset_y'].numpy()

            # Extract the region per patch that does not overlap with any other patch (i.e., no linear blending for
            # those pixels)
            ulx = batch['patch_valid_pixels_ulx'].numpy()
            uly = batch['patch_valid_pixels_uly'].numpy()
            lrx = batch['patch_valid_pixels_lrx'].numpy()
            lry = batch['patch_valid_pixels_lry'].numpy()

            # Inference
            y_pred = model(batch['input'].to(device)).cpu()
            y_pred = data_normalization.denormalize_numpy(y_pred, batch['dsm_mean'], batch['dsm_std'])

            # Iterate over each sample (DSM patch) within the batch
            for i in range(batch['input'].shape[0]):
                # Upper-left image coordinate of the i.th patch
                x = offsets_x[i]
                y = offsets_y[i]

                # Get blending weights
                weights = _get_blend_weights(tile_size, stride, ulx[i], uly[i], lrx[i], lry[i])

                # Perform blending
                weighted_tile = y_pred[i, 0, :, :] * weights
                raster_out[y:y+tile_size, x:x+tile_size] += weighted_tile

    return raster_out


def _get_blend_weights(tile_size, stride, ulx, uly, lrx, lry):
    """
    Returns the weights of a single tile for linear blending.

    :param tile_size:   int, tile size in pixels
    :param stride:      int, stride of the patches in pixels
    :param ulx:         int, upper-left x-coordinate, patch region that does not overlap with any other tile
                        (i.e., blending weight equals 1 for pixels within this region)
    :param uly:         int, upper-left y-coordinate, patch region that does not overlap with any other tile
                        (i.e., blending weight equals 1 for pixels within this region)
    :param lrx:         int, lower-right x-coordinate, patch region that does not overlap with any other tile
                        (i.e., blending weight equals 1 for pixels within this region)
    :param lry:         int, lower-right y-coordinate, patch region that does not overlap with any other tile
                        (i.e., blending weight equals 1 for pixels within this region)
    :return:
    """

    # Initialize the blending weights
    weights = np.ones((tile_size, tile_size))

    # Weights within the overlap region
    overlap = tile_size - stride
    linear_ramp = np.linspace(0, 1, overlap, endpoint=True)

    # Blending weights: left edge
    if ulx > 0:
        if ulx == overlap:
            weights[:, 0:ulx] *= np.tile(linear_ramp, (tile_size, 1))
        else:
            weights[:, ulx-overlap:ulx] *= np.tile(linear_ramp, (tile_size, 1))
            weights[:, 0:ulx-overlap] = 0

    # Blending weights: right edge
    if lrx < tile_size - 1:
        weights[:, lrx+1:] *= np.tile(np.flip(linear_ramp), (tile_size, 1))

    # Blending weights: top edge
    if uly > 0:
        if uly == overlap:
            n_pixels = uly
            weights[0:uly, :] *= np.tile(linear_ramp.reshape(n_pixels, 1), (1, tile_size))
        else:
            n_pixels = overlap
            weights[uly - overlap:uly, :] *= np.tile(linear_ramp.reshape(n_pixels, 1), (1, tile_size))
            weights[0:uly - overlap, :] = 0

    # Blending weights: bottom edge
    if lry < tile_size - 1:
        n_pixels = tile_size - lry - 1
        weights[lry + 1:, :] *= np.tile(np.flip(linear_ramp).reshape(n_pixels, 1), (1, tile_size))

    return weights
