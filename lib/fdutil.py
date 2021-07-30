import os


def make_dir(directory):
    """
    Creates a directory.

    :param directory:  str, directory path
    """

    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError:
            if not os.path.isdir(directory):
                raise
        else:
            print('Created directory {}'.format(directory))


def file_extension(filepath):
    """
    Returns the file extension of the given file.

    :param filepath:    str, path of a file
    :return:            str, file extension
    """

    return os.path.splitext(filepath)[1]


def filename(filepath):
    """
    Returns the name of the given file with extension.

    :param filepath:    str, path of a file
    :return:            str, filename with extension
    """

    return os.path.basename(filepath)


def filename_wo_ext(filepath):
    """
    Returns the name of the given file without extension.

    :param filepath:    str, path of a file
    :return:            str, filename without extension
    """

    name = os.path.splitext(os.path.basename(filepath))[0]
    return name


def file_exists(filepath):
    """
    Checks if the given file exists.

    :param filepath:    str, path of a file
    :return:            bool, True if the file exists, False otherwise
    """

    return os.path.exists(filepath)
