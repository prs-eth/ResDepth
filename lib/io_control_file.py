import pickle
from lib import fdutil, utils


def read_normalization_params_from_file(filepath):
    """
    Imports the normalization parameters from a pickle file.

    :param filepath:    str, path of the pickle file
    :return:            dict, imported pickle file
    """

    with open(filepath, 'rb') as fid:
        params = pickle.load(fid)

    return params


def write_normalization_params_to_file(filepath, mean, std):
    """
    Stores the normalization parameters as a pickle file.

    :param filepath:    str, path of the pickle file
    :param mean:        float, mean value (if None: local centering of individual data patches)
    :param std:         float, standard deviation
    """

    params = {
        'mean': mean,
        'std': std
    }

    with open(filepath, 'wb') as fid:
        pickle.dump(params, fid, protocol=pickle.HIGHEST_PROTOCOL)


def read_imagelist_from_file(file):
    """
    Reads a text file that lists the paths of a set of images (one line per image).

    :param file:  str, path of the text file
    :return:      list of strings, paths of the images
    """

    with open(file, 'r') as fid:
        image_list = fid.read().splitlines()

    return image_list


def read_pairlist_from_file(file_imagelist, file_pairlist, logger=None):
    """
    Reads a list of image pairs from a text file. Each line in the text file file_pairlist consists of a
    comma-separated list of image names, where each line specifies
    a) a single image (ResDepth-mono)
    b) a single image stereo pair (ResDepth-stereo)
    c) a single image pair consisting of three or more images
    d) multiple image stereo pairs (generalized ResDepth-stereo variant)

    The image paths are given in the text file file_imagelist (one line per image).

    :param file_imagelist:  str, path of the text file that lists the paths of the ortho-rectified images
    :param file_pairlist:   str, path of the text file that specifies the image pair(s)
    :param logger:          logger instance
    :return image_list:     list of strings, paths of the images
    :return image_pairs:    list of tuples (equal length of each tuple), each tuple consists of a single integer
                            (mono guidance) or n integers (with n >= 2) to define image pairs. Each integer specifies
                            an image index w.r.t. the images listed in image_list. image_pairs is None if an error
                            has been detected in file_pairlist.
    """

    if logger is None:
        logger = utils.setup_logger('read_pairlist_from_file', log_to_console=True, log_file=None)

    # Read the image list
    image_list = read_imagelist_from_file(file_imagelist)

    # Read the image pair list (pairs specified by image name)
    image_pairs_names = []
    with open(file_pairlist, 'r') as fid:
        line = fid.readline()

        while line:
            image_pairs_names.append(line.splitlines()[0].split(', '))
            line = fid.readline()

    # Verify that each image pair has the same length (i.e., equal number of images per pair)
    if len(set(map(len, image_pairs_names))) not in (0, 1):
        logger.error(f'Varying number of images per image pair detected in {file_pairlist}.\n')
        image_pairs = None
    else:
        # Convert the pair list from a list of image names to a list of tuples
        # (pairs specified by image indices w.r.t. image_list)
        image_pairs = []
        for pair in image_pairs_names:
            indices = []
            for image in pair:
                index = [i for i, elem in enumerate(image_list) if image in elem]

                if len(index) > 1:
                    logger.error(f'Found the image {image} multiple times in {file_imagelist}.\n')
                    return image_list, None

                elif len(index) == 0:
                    logger.error(f'The image {image} is not listed in {file_imagelist}.\n')
                    return image_list, None

                elif index in indices:
                    logger.error(f'Found the image {image} multiple times within the same image pair '
                                 f'in {file_imagelist}.\n')
                    return image_list, None
                else:
                    indices.append(index[0])

            if tuple(indices) in image_pairs:
                logger.error(f'Found the image pair {tuple(indices)} multiple times in {file_imagelist}.')
                for index in indices:
                    logger.info(f'Image {index}:\t{fdutil.filename(image_list[index])}')
                return image_list, None
            else:
                image_pairs.append(tuple(indices))

    return image_list, image_pairs
