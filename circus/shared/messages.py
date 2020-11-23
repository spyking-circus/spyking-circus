from colorama import Fore
import logging
# import os
import sys


def get_header():

    import circus
    version = "(" + circus.__version__ + ')'

    title = '#####             Welcome to the SpyKING CIRCUS              #####'
    nb_version = len(version)
    nb_blank = 55 - nb_version 
    white_spaces = ' '*(nb_blank //2)

    nb_blank_2 = 66 - 10 - len(white_spaces) - len(version)
    white_spaces_2 = ' '*nb_blank_2

    version = '#####'+ white_spaces + version + white_spaces_2 + '#####'

    header = '''
##################################################################
%s
%s
#####             Written by P.Yger and O.Marre              #####
##################################################################

''' %(title, version)

    return header


class InfoFilter(logging.Filter):
    def filter(self, rec):
        return rec.levelno == logging.DEBUG


def init_logging(logfile, debug=True, level=None):
    """
    Simple configuration of logging.
    """

    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    # allow user to override exact log_level
    if level:
        log_level = level

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
        filename=logfile,
        filemode='a'
    )

    return


def write_to_logger(logger, to_write, level='info'):
    for line in to_write:
        if level == 'info':
            logger.info(line)
        elif level in ['debug', 'default']:
            logger.debug(line)
        elif level == 'warning':
            logger.warning(line)


def print_and_log(to_print, level='info', logger=None, display=True):
    if display:
        if level == 'default':
            for line in to_print:
                print(Fore.WHITE + line + '\r')
        if level == 'info':
            print_info(to_print)
        elif level == 'error':
            print_error(to_print)

    if logger is not None:
        write_to_logger(logger, to_print, level)

    sys.stdout.flush()


def print_info(lines):
    """Prints informations messages, enhanced graphical aspects."""
    print(Fore.YELLOW + "-------------------------  Informations  -------------------------\r")
    for line in lines:
        print(Fore.YELLOW + "| " + line + '\r')
    print(Fore.YELLOW + "------------------------------------------------------------------\r" + Fore.WHITE)


def print_error(lines):
    """Prints errors messages, enhanced graphical aspects."""
    print(Fore.RED + "----------------------------  Error  -----------------------------\r")
    for line in lines:
        print(Fore.RED + "| " + line + '\r')
    print(Fore.RED + "------------------------------------------------------------------\r" + Fore.WHITE)


def get_colored_header():
    return Fore.GREEN + get_header() + Fore.RESET
