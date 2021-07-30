import numpy as np
from osgeo import gdal
from scipy import ndimage


def load_raster(fn, mode=gdal.GA_ReadOnly):
    """
    Loads a GeoTiff raster file as a gdal dataset.

    :param fn:      str, path to the GeoTiff raster file
    :param mode:    gdal reading mode
    :return:        gdal.Dataset instance
    """

    ds = gdal.Open(fn, mode)

    if ds is None:
        raise ValueError('Could not open {}'.format(fn))

    return ds


def load_mask_raster(file):
    """
    Loads a GeoTiff raster file as a boolean mask.

    :param file:                str, path to the GeoTiff raster file
    :return mask_building:      np.array (boolean), mask raster
    :return mask_nodata:        float, nodata value of the raster
    """

    if isinstance(file, gdal.Dataset):
        raster = file.ReadAsArray()
        nodata = file.GetRasterBand(1).GetNoDataValue()
    else:
        ds = load_raster(file)
        raster = ds.ReadAsArray()
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None

    # Convert to a boolean mask
    mask_building = np.zeros_like(raster, dtype=np.bool)
    mask_building[raster == 1] = True

    # Mask out nodata pixels
    mask_nodata = raster == nodata
    mask_building[mask_nodata] = False

    return mask_building, mask_nodata


def get_raster_extent(fn):
    """
    Reads the spatial extent of a GeoTiff raster file.

    :param fn:  string or gdal.Dataset instance, path to the GeoTiff raster file or imported GeoTiff raster file
    :return:    dict, dictionary consisting of the following key-value pairs (units specified by geotransform):
                minX: float, minimum X coordinate
                maxX: float, maximum X coordinate
                minY: float, minimum Y coordinate
                maxY: float, maximum Y coordinate
                cols: int, number of columns
                rows: int, number of rows
                gsdX: positive float, GSD (ground sampling distance) in X-direction
                gsdY: positive (!) float, GSD (ground sampling distance) in Y-direction
    """

    if isinstance(fn, gdal.Dataset):
        ds = fn
    else:
        ds = load_raster(fn)

    gt = ds.GetGeoTransform()

    cols = ds.RasterXSize
    rows = ds.RasterYSize

    minX = gt[0]
    [maxX, minY] = gdal.ApplyGeoTransform(gt, cols, rows)
    maxY = gt[3]

    return {
        'minX': minX, 'maxX': maxX, 'minY': minY, 'maxY': maxY,
        'cols': cols, 'rows': rows, 'gsdX': gt[1], 'gsdY': -gt[5]
    }


def dilate_mask(mask_in, iterations=1):
    """
    Dilates a binary mask.

    :param mask_in:     np.array, binary mask to be dilated
    :param iterations:  int, number of dilation iterations
    :return:            np.array, dilated binary mask
    """

    return ndimage.morphology.binary_dilation(mask_in, iterations=iterations)


def create_regular_grid(area_defn, tile_size, stride=None):
    """
    Defines a regular grid of (overlapping) tiles.

    :param area_defn:       dictionary, defines one or multiple rectangularly-shaped geographic regions from which
                            DSM patches will be sampled. The dictionary is composed of the following key-value pairs:
                            x_extent:   list of n tuples, where n denotes the number of rectangular regions (stripes).
                                        Each tuple defines the upper-left and lower-right x-coordinate of a rectangular
                                        region (stripe).
                            y_extent:   list of n tuples, where n denotes the number of rectangular regions (stripes).
                                        Each tuple defines the upper-left and lower-right y-coordinate of a rectangular
                                        region (stripe).

                            Assumption: The i.th tuple of x_extent and i.th tuple of y_extent define a geographically
                                        rectangular region (stripe).

    :param tile_size:       int, tile size in pixels
    :param stride:          int, stride in pixels (if None: stride equals tile_size)

    :return tile_position:      list of tuples, i.th tuple (uly, ulx) specifies the upper-left image coordinates of the
                                i.th tile (w.r.t. the full raster)
    :return region_wo_overlap:  list of tuples, i.th tuple (uly, ulx, lry, lrx) specifies the pixels of the i.th tile
                                that do not overlap with any other tile
    """

    if stride is None:
        stride = tile_size

    tile_position = []
    region_wo_overlap = []

    # Number of regions specified in area_defn
    num_regions = len(area_defn['x_extent'])

    # Iterate over each region
    for i in range(num_regions):
        # Extent of the i.th region
        x = area_defn['x_extent'][i]
        y = area_defn['y_extent'][i]

        # Upper-left coordinates of the tile (w.r.t. full raster)
        uly = y[0]
        lry = y[0]

        # Initialization
        border_uly = 0
        border_lry = stride - 1

        # Split the i.th region into a grid of regular tiles
        while lry < y[1]:
            # Initialization
            ulx = x[0]
            lrx = x[0]
            border_ulx = 0
            border_lrx = stride - 1

            # Compute the lower-right y-coordinate of the tile
            lry = uly + tile_size - 1

            # Check if the tile overlaps the region (in y-direction): if yes, shift the tile upwards such that its
            # lower border coincides with the lower border of the region
            if lry >= y[1]:
                border_uly += lry - y[1]
                lry = y[1]
                uly = y[1] - tile_size + 1
                border_lry = tile_size - 1

            while lrx < x[1]:
                # Compute lower-right x-coordinate of the tile
                lrx = ulx + tile_size - 1

                # Check if the tile overlaps the area (in x-direction): if yes, shift the tile to the left such that
                # its right border coincides with the right border of the region
                if lrx >= x[1]:
                    border_ulx += lrx - x[1]
                    lrx = x[1]
                    ulx = x[1] - tile_size + 1
                    border_lrx = tile_size - 1

                # Save the upper-left corner coordinates of the tile
                tile_position.append((int(uly), int(ulx)))

                # Save the pixels of the tile that do not overlap with any other tile
                region_wo_overlap.append((int(border_uly), int(border_ulx), int(border_lry), int(border_lrx)))

                ulx += stride
                border_ulx = tile_size - stride

            uly += stride
            border_uly = tile_size - stride

    return tile_position, region_wo_overlap


def export_data_as_raster(in_ds, filepath, data, offset_x, offset_y, data_type=None, nodata=None,
                          flag_stats=True, compress=True):
    """
    Exports a np.array as GeoTiff.

    :param in_ds:           gdal.Dataset or str, initial DSM loaded as gdal.Dataset
                            (or alternatively, path to the initial DSM GeoTiff raster); used to copy the projection and
                            geotransform
    :param filepath:        str, path to the output raster file
    :param data:            np.array, DSM to be exported
    :param offset_x:        int, x-coordinate offset of data w.r.t. in_ds (will be applied to the geotransform of in_ds)
    :param offset_y:        int, y-coordinate offset of data w.r.t. in_ds (will be applied to the geotransform of in_ds)
    :param data_type:       GDALDataType (if None: same GDALDataType as in_ds)
    :param nodata:          float or int, nodata value of the output raster file (if None: same nodata value as in_ds)
    :param flag_stats:      boolean, True to compute band statistics, False otherwise
    :param compress:        boolean, True to export data using 'COMPRESS=LZW' compression, False otherwise
    """

    if not isinstance(in_ds, gdal.Dataset):
        in_ds = load_raster(in_ds)

    # Determine the data type
    if data_type is None:
        data_type = in_ds.GetRasterBand(1).DataType

    if data.ndim == 2:
        data = np.expand_dims(data, axis=2)

    # Number of bands of the input raster
    num_bands = data.shape[2]

    # Create output file
    driver = gdal.GetDriverByName('GTiff')
    if compress:
        out_ds = driver.Create(filepath, data.shape[1], data.shape[0], num_bands, data_type, options=['COMPRESS=LZW'])
    else:
        out_ds = driver.Create(filepath, data.shape[1], data.shape[0], num_bands, data_type)

    # Set the projection of the output file
    out_ds.SetProjection(in_ds.GetProjection())

    # Set the geotransform of the output file
    gt = in_ds.GetGeoTransform()
    subset_ulx, subset_uly = gdal.ApplyGeoTransform(gt, offset_x, offset_y)  # convert offset to world coordinates
    out_gt = list(gt)
    out_gt[0] = subset_ulx  # update offset of the geotransform
    out_gt[3] = subset_uly
    out_ds.SetGeoTransform(out_gt)

    # Write data to the output file
    for b in range(1, num_bands + 1):
        out_band = out_ds.GetRasterBand(b)
        out_band.WriteArray(data[:, :, b - 1])

    out_band.FlushCache()

    # Set the nodata value
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    else:
        out_band.SetNoDataValue(in_ds.GetRasterBand(1).GetNoDataValue())

    # Compute raster band statistics
    out_band.ComputeBandStats(flag_stats)

    # Close output tile
    out_ds = None
    out_band = None
