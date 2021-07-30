from lib import arguments


def all_keys_known(dictionary, known_keys, logger=None):
    """
    Checks whether all keys of the dictionary are listed in the list of known keys.

    :param dictionary:   dict, dictionary whose keys are verified
    :param known_keys:   list, list of known keys
    :param logger:       logger instance
    :return:             boolean, True if all keys of the dictionary are listed in known_keys, False otherwise
    """

    unknown_keys = [k for k in dictionary if k not in known_keys]

    if unknown_keys:
        if logger:
            logger.error('The following keys are unknown: {}.\n'.
                         format(','.join(["'{}'".format(x) for x in unknown_keys])))
        else:
            print('ERROR: The following keys are unknown: {}.\n'.
                  format(','.join(["'{}'".format(x) for x in unknown_keys])))
        return False
    else:
        return True


def is_boolean(value, arg_name, logger=None):
    """
    Verifies whether a parameter is correctly defined as boolean.

    :param value:      value of the parameter
    :param arg_name:   str, parameter name
    :param logger:     logger instance
    :return:           boolean, True if value is a boolean, False otherwise
    """

    if not isinstance(value, bool):
        if logger:
            logger.error(f"Invalid value for the argument '{arg_name}': {value}. Specify a boolean.\n")
        else:
            print(f"ERROR: Invalid value for the argument '{arg_name}': {value}. Specify a boolean.\n")
        return False
    else:
        return True


def is_string(value, arg_name, logger=None):
    """
    Verifies whether a parameter is correctly defined as string.

    :param value:      value of the parameter
    :param arg_name:   str, parameter name
    :param logger:     logger instance
    :return:           boolean, True if value is a string, False otherwise
    """

    if not isinstance(value, str):
        if logger:
            logger.error(f"Invalid value for the argument '{arg_name}': {value}. Specify a string.\n")
        else:
            print(f"ERROR: Invalid value for the argument '{arg_name}': {value}. Specify a string.\n")
        return False
    else:
        return True


def is_positive_integer(value, arg_name, logger=None, zero_allowed=False):
    """
    Verifies whether a parameter is correctly defined as a positive integer.

    :param value:           value of the parameter
    :param arg_name:        str, parameter name
    :param logger:          logger instance
    :param zero_allowed:    boolean, True if zero is valid for value, False otherwise
    :return:                boolean, True if value is a positive integer (including zero according to the flag
                            zero_allowed), False otherwise
    """

    if zero_allowed:
        if type(value) is not int or value < 0:
            if logger:
                logger.error(f"Invalid value for the argument '{arg_name}': {value}. Specify an integer >= 0.\n")
            else:
                print(f"ERROR: Invalid value for the argument '{arg_name}': {value}. Specify an integer >= 0.\n")
            return False
        else:
            return True
    else:
        if type(value) is not int or value <= 0:
            if logger:
                logger.error(f"Invalid value for the argument '{arg_name}': {value}. Specify a positive integer.\n")
            else:
                logger.error(f"ERROR: Invalid value for the argument '{arg_name}': {value}. "
                             "Specify a positive integer.\n")
            return False
        else:
            return True


def valid_act_fn(value, identifier, arg_name, logger=None):
    """
    Verifies whether the activation function is valid.

    :param value:        str, name of the activation function
    :param identifier:   str, activation function identifier
    :param arg_name:     str, parameter name of the activation function according to the configuration file
    :param logger:       logger instance
    :return:             boolean, True if the activation function is valid, False otherwise
    """

    if value not in arguments.ACTIVATION_FUNCTIONS:
        if logger:
            logger.error(f"Invalid activation function of the {identifier}: {value}. Choose among "
                         f"{arguments.ACTIVATION_FUNCTIONS} to specify {arg_name}.\n")
        else:
            print(f"ERROR: Invalid activation function of the {identifier}: {value}. Choose among "
                  f"{arguments.ACTIVATION_FUNCTIONS} to specify {arg_name}.\n")
        return False
    else:
        return True


def valid_allocation(value, logger=None):
    """
    Verifies whether the data allocation strategy is valid.

    :param value:        str, name of the data allocation strategy
    :param logger:       logger instance
    :return:             boolean, True if the data allocation strategy is valid, False otherwise
    """

    if value not in arguments.ALLOCATION_STRATEGIES:
        if logger:
            logger.error(f"Invalid allocation strategy: '{value}'. Choose among {arguments.ALLOCATION_STRATEGIES}.\n")
        else:
            print(f"ERROR: Invalid allocation strategy: '{value}'. Choose among {arguments.ALLOCATION_STRATEGIES}.\n")
        return False
    else:
        return True


def valid_tile_size(value, arg_name, min_power=4, logger=None):
    """
    Verifies that the tile size is defined as an integer in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096].

    :param value:     int, tile size
    :param arg_name:  str, parameter name
    :param min_power: int, 2^min_power as minimum tile size (consistency with the number of downsampling layers)
    :param logger:    logger instance
    :return:          boolean, True if the tile size is correctly defined, False otherwise
    """

    error = False

    if not isinstance(value, int):
        if logger:
            logger.error(f"Invalid value for the argument {arg_name}: {value}. Enter an integer.\n")
        else:
            print(f"ERROR: Invalid value for the argument {arg_name}: {value}. Enter an integer.\n")
        error = True
    if value not in [2 ** i for i in range(min_power, 12)]:
        if logger:
            logger.error(f"Invalid value for the argument {arg_name}: {value}. Choose among "
                         f"{[2 ** i for i in range(min_power, 12)]}.\n")
        else:
            print(f"ERROR: Invalid value for the argument {arg_name}: {value}. Choose among "
                  f"{[2 ** i for i in range(min_power, 12)]}.\n")
        error = True

    return not error
