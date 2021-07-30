import numpy as np
import sys

from lib import fdutil, rasterutils


STRATEGIES = ['5-crossval_vertical', '5-crossval_horizontal']


def _verify_inputs(fn_raster_in, allocation_strategy, test_stripe, crossval_training):
    """
    Verifies the inputs of the function allocate_data().

    :param fn_raster_in:          str, path to the GeoTiff raster file
    :param allocation_strategy:   str, allocation strategy (see parameter STRATEGIES)
    :param test_stripe:           int, index of the test stripe (or validation stripe if cross-validation is enabled)
    :param crossval_training:     bool, True if the raster is used for cross-validation
                                  (split into training and validation regions only), False otherwise
    """

    # Check that the input GeoTiff raster exists
    if not fdutil.file_exists(fn_raster_in):
        print('Input raster does not exist: {}'.format(fn_raster_in))
        sys.exit(1)

    if not isinstance(test_stripe, int):
        print("'test_stripe' must be an integer in the range [0,4].")
        sys.exit(1)

    if test_stripe > 4:
        print("'test_stripe' must be an integer in the range [0,4].")
        sys.exit(1)

    if allocation_strategy not in STRATEGIES:
        print("{} as 'allocation_strategy' is not a valid choice. Choose among: {}.".format(allocation_strategy,
                                                                                            STRATEGIES))
        sys.exit(1)

    if not isinstance(crossval_training, bool):
        print("'crossval_training' must be boolean.")
        sys.exit(1)


def allocate_data(fn_raster_in, allocation_strategy, test_stripe=0, crossval_training=False):
    """
    Splits a given raster into geographically separate stripes for training, validation, and testing.
    Assumption: the validation stripe is located to the right/bottom (east/south) of the test stripe (cyclic order).

    :param fn_raster_in:          str, path to the GeoTiff raster file
    :param allocation_strategy:   str, allocation strategy (see parameter STRATEGIES)
    :param test_stripe:           int, index of the test stripe (or validation stripe if cross-validation is enabled)
    :param crossval_training:     bool, True if the raster is used for cross-validation
                                  (split into training and validation regions only), False otherwise
    :return:                      returns three dictionaries train, val, and test, where each dictionary defines
                                  geographically rectangular regions. Each dictionary is composed of the following
                                  key-value pairs:
                                  x_extent:   list of n tuples, where n denotes the number of rectangular regions
                                              (stripes). Each tuple defines the upper-left and lower-right
                                              x-coordinate of a rectangular region (stripe).
                                  y_extent:   list of n tuples, where n denotes the number of rectangular regions
                                              (stripes). Each tuple defines the upper-left and lower-right
                                              y-coordinate of a rectangular region (stripe).

                                  Assumption: The i.th tuple of x_extent and i.th tuple of y_extent define a
                                              geographically rectangular region (stripe).
    """

    # Check inputs
    _verify_inputs(fn_raster_in, allocation_strategy, test_stripe, crossval_training)

    if allocation_strategy == '5-crossval_vertical':
        train, val, test = _allocate_5crossval_vertical(fn_raster_in, test_stripe, crossval_training)

    elif allocation_strategy == '5-crossval_horizontal':
        train, val, test = _allocate_5crossval_horizontal(fn_raster_in, test_stripe, crossval_training)

    return train, val, test


def _allocate_5crossval_vertical(fn_raster_in, test_stripe, crossval_training):
    """
    Splits the geographic area of fn_raster_in into five equally large and mutually exclusive vertical stripes
    (north-south oriented) for training, validation, and testing (or training and validation only if cross-validation
    is enabled). Assumption: the validation stripe is located to the right (east) of the test stripe (cyclic order).

    :param fn_raster_in:        str, path to the GeoTiff raster file
    :param test_stripe:         int, index of the test stripe (or validation stripe if cross-validation is enabled)
    :param crossval_training:   bool, True if the raster is used for cross-validation
                                (split into training and validation regions only), False otherwise
    :return:                    returns three dictionaries train, val, and test, where each dictionary defines
                                geographically rectangular regions (vertically oriented stripes). Each dictionary is
                                composed of the following key-value pairs:
                                x_extent:   list of n tuples, where n denotes the number of rectangular regions
                                            (stripes). Each tuple defines the upper-left and lower-right
                                            x-coordinate of a rectangular region (stripe).
                                y_extent:   list of n tuples, where n denotes the number of rectangular regions
                                            (stripes). Each tuple defines the upper-left and lower-right
                                            y-coordinate of a rectangular region (stripe).

                                Assumption: The i.th tuple of x_extent and i.th tuple of y_extent define a
                                            geographically rectangular region (stripe).
    """

    # Get the extent of the input raster
    extent = rasterutils.get_raster_extent(fn_raster_in)
    cols = extent['cols']
    rows = extent['rows']

    # Compute the width of the stripes
    width = int(round(float(cols) * 0.2))

    # Compute the extent in X-direction of the stripes
    x_start = 0
    x_extent = []

    for i in range(5):
        if i < 4:
            x_end = x_start + width - 1
        else:
            x_end = cols - 1

        x_extent.append((x_start, x_end))
        x_start = x_end + 1

    # Validation and test stripe: compute the extent in Y-direction
    y_val = [(0, rows - 1)]
    y_test = [(0, rows - 1)]

    if crossval_training is False:
        if test_stripe == 0:
            # Stripe order:  | test | val | train | train | train |
            x_train = [(x_extent[2][0], x_extent[4][1])]
            x_val = [x_extent[1]]
            x_test = [x_extent[0]]
            y_train = [(0, rows - 1)]

        elif test_stripe == 1:
            # Stripe order:  | train | test | val | train | train |
            x_train = [x_extent[0], (x_extent[3][0], x_extent[4][1])]
            x_val = [x_extent[2]]
            x_test = [x_extent[1]]
            y_train = [(0, rows - 1), (0, rows - 1)]

        elif test_stripe == 2:
            # Stripe order:  | train | train | test | val | train |
            x_train = [(x_extent[0][0], x_extent[1][1]), x_extent[4]]
            x_val = [x_extent[3]]
            x_test = [x_extent[2]]
            y_train = [(0, rows - 1), (0, rows - 1)]

        elif test_stripe == 3:
            # Stripe order:  | train | train | train | test | val |
            x_train = [(x_extent[0][0], x_extent[2][1])]
            x_val = [x_extent[4]]
            x_test = [x_extent[3]]
            y_train = [(0, rows - 1)]

        elif test_stripe == 4:
            # Stripe order:  | val | train | train | train | test |
            x_train = [(x_extent[1][0], x_extent[3][1])]
            x_val = [x_extent[0]]
            x_test = [x_extent[4]]
            y_train = [(0, rows - 1)]

        test = {'x_extent': x_test, 'y_extent': y_test}

    else:
        if test_stripe == 0:
            # Stripe order:  | val | train | train | train | train |
            x_train = [(x_extent[1][0], x_extent[4][1])]
            x_val = [x_extent[0]]
            y_train = [(0, rows - 1)]

        elif test_stripe == 1:
            # Stripe order:  | train | val | train | train | train |
            x_train = [x_extent[0], (x_extent[2][0], x_extent[4][1])]
            x_val = [x_extent[1]]
            y_train = [(0, rows - 1), (0, rows - 1)]

        elif test_stripe == 2:
            # Stripe order:  | train | train | val | train | train |
            x_train = [(x_extent[0][0], x_extent[1][1]), (x_extent[3][0], x_extent[4][1])]
            x_val = [x_extent[2]]
            y_train = [(0, rows - 1), (0, rows - 1)]

        elif test_stripe == 3:
            # Stripe order:  | train | train | train | val | train |
            x_train = [(x_extent[0][0], x_extent[2][1]), x_extent[4]]
            x_val = [x_extent[3]]
            y_train = [(0, rows - 1), (0, rows - 1)]

        elif test_stripe == 4:
            # Stripe order:  | train | train | train | train | val |
            x_train = [(x_extent[0][0], x_extent[3][1])]
            x_val = [x_extent[4]]
            y_train = [(0, rows - 1)]

        test = {}

    train = {'x_extent': x_train, 'y_extent': y_train}
    val = {'x_extent': x_val, 'y_extent': y_val}

    return train, val, test


def _allocate_5crossval_horizontal(fn_raster_in, test_stripe, crossval_training):
    """
    Splits the geographic area of fn_raster_in into five equally large and mutually exclusive horizontal stripes
    (west-east oriented) for training, validation, and testing (or training and validation only if cross-validation
    is enabled). Assumption: the validation stripe is located to the bottom (south) of the test stripe (cyclic order).

    :param fn_raster_in:        str, path to the GeoTiff raster file
    :param test_stripe:         int, index of the test stripe (or validation stripe if cross-validation is enabled)
    :param crossval_training:   bool, True if the raster is used for cross-validation
                                (split into training and validation regions only), False otherwise
    :return:                    returns three dictionaries train, val, and test, where each dictionary defines
                                geographically rectangular regions (horizontally oriented stripes). Each dictionary is
                                composed of the following key-value pairs:
                                x_extent:   list of n tuples, where n denotes the number of rectangular regions
                                            (stripes). Each tuple defines the upper-left and lower-right
                                            x-coordinate of a rectangular region (stripe).
                                y_extent:   list of n tuples, where n denotes the number of rectangular regions
                                            (stripes). Each tuple defines the upper-left and lower-right
                                            y-coordinate of a rectangular region (stripe).

                                Assumption: The i.th tuple of x_extent and i.th tuple of y_extent define a
                                            geographically rectangular region (stripe).
    """

    # Get the extent of the input raster
    extent = rasterutils.get_raster_extent(fn_raster_in)
    cols = extent['cols']
    rows = extent['rows']

    # Compute the height of the stripes
    height = int(round(float(rows) * 0.2))

    # Compute the extent in Y-direction of the stripes
    y_start = 0
    y_extent = []

    for i in range(5):
        if i < 4:
            y_end = y_start + height - 1
        else:
            y_end = rows - 1

        y_extent.append((y_start, y_end))
        y_start = y_end + 1

    # Validation and test stripe: compute the extent in X-direction
    x_val = [(0, cols - 1)]
    x_test = [(0, cols - 1)]

    if crossval_training is False:
        if test_stripe == 0:
            # Stripe order:  | test | val | train | train | train |
            y_train = [(y_extent[2][0], y_extent[4][1])]
            y_val = [y_extent[1]]
            y_test = [y_extent[0]]
            x_train = [(0, cols - 1)]

        elif test_stripe == 1:
            # Stripe order:  | train | test | val | train | train |
            y_train = [y_extent[0], (y_extent[3][0], y_extent[4][1])]
            y_val = [y_extent[2]]
            y_test = [y_extent[1]]
            x_train = [(0, cols - 1), (0, cols - 1)]

        elif test_stripe == 2:
            # Stripe order:  | train | train | test | val | train |
            y_train = [(y_extent[0][0], y_extent[1][1]), y_extent[4]]
            y_val = [y_extent[3]]
            y_test = [y_extent[2]]
            x_train = [(0, cols - 1), (0, cols - 1)]

        elif test_stripe == 3:
            # Stripe order:  | train | train | train | test | val |
            y_train = [(y_extent[0][0], y_extent[2][1])]
            y_val = [y_extent[4]]
            y_test = [y_extent[3]]
            x_train = [(0, cols - 1)]

        elif test_stripe == 4:
            # Stripe order:  | val | train | train | train | test |
            y_train = [(y_extent[1][0], y_extent[3][1])]
            y_val = [y_extent[0]]
            y_test = [y_extent[4]]
            x_train = [(0, cols - 1)]

        test = {'x_extent': x_test, 'y_extent': y_test}

    else:
        if test_stripe == 0:
            # Stripe order:  | val | train | train | train | train |
            y_train = [(y_extent[1][0], y_extent[4][1])]
            y_val = [y_extent[0]]
            x_train = [(0, cols - 1)]

        elif test_stripe == 1:
            # Stripe order:  | train | val | train | train | train |
            y_train = [y_extent[0], (y_extent[2][0], y_extent[4][1])]
            y_val = [y_extent[1]]
            x_train = [(0, cols - 1), (0, cols - 1)]

        elif test_stripe == 2:
            # Stripe order:  | train | train | val | train | train |
            y_train = [(y_extent[0][0], y_extent[1][1]), (y_extent[3][0], y_extent[4][1])]
            y_val = [y_extent[2]]
            x_train = [(0, cols - 1), (0, cols - 1)]

        elif test_stripe == 3:
            # Stripe order:  | train | train | train | val | train |
            y_train = [(y_extent[0][0], y_extent[2][1]), y_extent[4]]
            y_val = [y_extent[3]]
            x_train = [(0, cols - 1), (0, cols - 1)]

        elif test_stripe == 4:
            # Stripe order:  | train | train | train | train | val |
            y_train = [(y_extent[0][0], y_extent[3][1])]
            y_val = [y_extent[4]]
            x_train = [(0, cols - 1)]

        test = {}

    train = {'x_extent': x_train, 'y_extent': y_train}
    val = {'x_extent': x_val, 'y_extent': y_val}

    return train, val, test


def indices_from_area_defn(area_defn, tile_size):
    """
    Returns the location (upper-left image coordinates) of valid patch positions.

    :param area_defn:   dictionary, defines one or multiple rectangularly-shaped geographic regions from which
                        DSM patches will be sampled. The dictionary is composed of the following key-value pairs:
                        x_extent:   list of n tuples, where n denotes the number of rectangular regions (stripes).
                                    Each tuple defines the upper-left and lower-right x-coordinate of a rectangular
                                    region (stripe).
                        y_extent:   list of n tuples, where n denotes the number of rectangular regions (stripes).
                                    Each tuple defines the upper-left and lower-right y-coordinate of a rectangular
                                    region (stripe).

                        Assumption: The i.th tuple of x_extent and i.th tuple of y_extent define a geographically
                                    rectangular region (stripe).

    :param tile_size:   int, tile size in pixels,
    :return:            list of (y,x) tuples, upper-left image coordinates of valid patch positions. Note that the
                        returned patch positions do not exceed the area specified in area_defn.
   """

    # Initialize output list
    valid_positions = []

    # Number of regions specified in area_defn
    num_regions = len(area_defn['x_extent'])

    for i in range(num_regions):
        # Extent of the i.th region
        x = area_defn['x_extent'][i]
        y = area_defn['y_extent'][i]

        # Compute valid x-coordinates of the i.th region
        x_start = x[0]
        x_end = x[1] - tile_size + 1
        x_indices = np.linspace(x_start, x_end, x_end - x_start + 1, dtype=int)

        # Compute valid y-coordinates of the i.th region
        y_start = y[0]
        y_end = y[1] - tile_size + 1
        y_indices = np.linspace(y_start, y_end, y_end - y_start + 1, dtype=int)

        for y in y_indices:
            for x in x_indices:
                valid_positions.append((y, x))

    return valid_positions
