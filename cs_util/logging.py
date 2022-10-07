"""LOGGING.

:Description: This script contains utility methods for job execution
and progress logging.

:Author: Martin Kilbinger

:Author: Martin Kilbinge <martin.kilblinger@cea.fr>

"""


import sys


def log_command(argv, name=None, close_no_return=True):
    """Log Command.

    Write command with arguments to a file or stdout.
    Choose name = 'sys.stdout' or 'sys.stderr' for output on sceen.

    MKDEBUG copied from shapepipe:cfis

    Parameters
    ----------
    argv : list
        Command line arguments
    name : str
        Output file name (default: 'log_<command>')
    close_no_return : bool
        If True (default), close log file. If False, keep log file open
        and return file handler

    Returns
    -------
    filehandler
        log file handler (if close_no_return is False)

    """
    if name is None:
        name = 'log_' + os.path.basename(argv[0])

    if name == 'sys.stdout':
        f = sys.stdout
    elif name == 'sys.stderr':
        f = sys.stderr
    else:
        f = open(name, 'w')

    for a in argv:

        # Quote argument if special characters
        if '[' in a or ']' in a:
            a = f'\"{a}\"'

        print(a, end='', file=f)
        print(' ', end='', file=f)

    print('', file=f)

    if not close_no_return:
        return f

    if name != 'sys.stdout' and name != 'sys.stderr':
        f.close()
