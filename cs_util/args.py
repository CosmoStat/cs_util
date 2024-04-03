"""ARGS.

:Description: Handling of command line arguments.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>


"""

import sys
from optparse import OptionParser


def parse_options(p_def, short_options, types, help_strings, args=None):
    """Parse command line options.

    Parameters
    ----------
    p_def : dict
        default parameter values
    short_options : dict
        command line options short (one character) versions
    types : dict
        command line options types
    help_strings : dict
        command line options help strings
    args : list, optional
        list of arguments, default is None, in which case sys.args[1:]
        is used

    Returns
    -------
    options: tuple
        Command line options

    """
    if args is None:
        args = sys.argv[1:]

    usage = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    # Loop over default parameter values
    for key in p_def:
        # Process if help string exists
        if key in help_strings:
            # Set short option
            if key in short_options:
                short = short_options[key]
            else:
                # Default is no short option
                short = ""

            # Set type
            if key in types:
                typ = types[key]
            else:
                # Default is str
                typ = "string"

            # Special case: bool type
            if typ == "bool":
                parser.add_option(
                    f"{short}",
                    f"--{key}",
                    dest=key,
                    default=False,
                    action="store_true",
                    help=help_strings[key].format(p_def[key]),
                )
            else:
                parser.add_option(
                    short,
                    f"--{key}",
                    dest=key,
                    type=typ,
                    default=p_def[key],
                    help=help_strings[key].format(p_def[key]),
                )

    # Add verbose option
    parser.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help=f"verbose output",
    )

    options_values, my_args = parser.parse_args(args)

    # Transform parameter values to dict
    options = {}
    for key in vars(options_values):
        options[key] = getattr(options_values, key)

    # Add other default values which do not have argument option
    for key in p_def:
        if key not in options:
            options[key] = p_def[key]

    return options


def my_string_split(string, num=-1, verbose=False, stop=False, sep=None):
    """My String Split.

    Split a *string* into a list of strings. Choose as separator
    the first in the list [space, underscore] that occurs in the string.
    (Thus, if both occur, use space.)

    Parameters
    ----------
    string : str
        Input string
    num : int
        Required length of output list of strings, -1 if no requirement.
    verbose : bool
        Verbose output
    stop : bool
        Stop programs with error if True, return None and continues otherwise
    sep : bool
        Separator, try ' ', '_', and '.' if None (default)

    Raises
    ------
    ValueError
        If number of elements in string and num are different, for stop=True
        If no separator found in string

    Returns
    -------
    list
        List of string on success, and None if failed

    """
    if string is None:
        return None

    if sep is None:
        has_space = string.find(" ")
        has_underscore = string.find("_")
        has_dot = string.find(".")

        if has_space != -1:
            my_sep = " "
        elif has_underscore != -1:
            my_sep = "_"
        elif has_dot != -1:
            my_sep = "."
        else:
            # no separator found, does string consist of only one element?
            if num == -1 or num == 1:
                my_sep = None
            else:
                raise ValueError(
                    "No separator (' ', '_', or '.') found in string"
                    + f" '{string}', cannot split"
                )
    else:
        if not string.find(sep):
            raise ValueError(
                f"No separator '{sep}' found in string '{string}' cannot split"
            )
        my_sep = sep

    res = string.split(my_sep)

    if num != -1 and num != len(res) and stop:
        raise ValueError(
            f"String '{len(res)}' has length {num}, required is {num}"
        )

    return res
