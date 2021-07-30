import collections
from copy import deepcopy
from easydict import EasyDict as edict
import json


def read_json(file):
    """
    Reads a json configuration file.

    :param file:  str, path of the json file
    :return:      EasyDict, imported json file
    """

    try:
        with open(file) as f:
            cfg = json.load(f)
    except json.JSONDecodeError:
        cfg = {}
        print(f'ERROR: Cannot read the file: {file}')

    return edict(cfg)


def write_json(data, outfile):
    """
    Writes the dictionary data to a json file.

    :param data:     dictionary, data to be stored as json file
    :param outfile:  str, path of the output file
    """

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=2)


def print_json(cfg, sort_keys=False, logger=None):
    """
    Prints a json configuration file to the console.

    :param cfg:         string or Dict/EasyDict, path of the json file or file imported as dictionary returned by the
                        function read_json().
    :param sort_keys:   boolean, True if the keys of the dictionary should be sorted, False otherwise
    :param logger:      logger instance
    """

    if not isinstance(cfg, dict):
        cfg = read_json(cfg)

    if logger:
        logger.info(json.dumps(cfg, indent=4, sort_keys=sort_keys))
    else:
        print(json.dumps(cfg, indent=4, sort_keys=sort_keys))


def merge(cfg_default, cfg_user):
    """
    Returns a new dictionary by recursively merging the dictionary cfg_user into the dictionary cfg_default. I.e.,
    the output dictionary is a deep copy of cfg_default with updated values and additional key-value pairs from
    cfg_user.

    :param cfg_default:  Dict or EasyDict, dictionary containing the default settings
    :param cfg_user:     Dict or EasyDict, dictionary containing the user-specific settings
    :return:             EasyDict, merged dictionary
    """

    result = deepcopy(cfg_default)

    for key, value in cfg_user.items():
        if isinstance(value, collections.Mapping):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(cfg_user[key])

    return edict(result)


def remove_obsolete_keys(cfg):
    """
    Removes obsolete keys from the config dictionary (modified in-place).

    :param cfg:   EasyDict, json user configuration file imported as dictionary
    """

    if not isinstance(cfg, edict):
        cfg = edict(cfg)

    # Remove multi-view settings
    if cfg.model.input_channels != 'geom-multiview':
        cfg.pop('multiview', None)
