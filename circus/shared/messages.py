from colorama import Fore
import sys, os, logging

def get_header():

    import circus
    version = circus.__version__

    if len(version) == 3:
        title = '#####            Welcome to the SpyKING CIRCUS (%s)         #####' %version
    elif len(version) == 5:
        title = '#####           Welcome to the SpyKING CIRCUS (%s)        #####' %version

    header = '''
##################################################################
%s
#####                                                        #####
#####              Written by P.Yger and O.Marre             #####
##################################################################

''' %title

    return header

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

    logging.basicConfig(level=log_level,
                        format='%(asctime)s %(levelname)-8s [%(name)s] %(message)s',
                        filename=logfile,
                        filemode='a')
    return logging.getLogger("circus")


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
                print Fore.WHITE + line + '\r'
        if level == 'info':
            print_info(to_print)
        elif level == 'error':
            print_error(to_print)

    if logger is not None:
        write_to_logger(logger, to_print, level)

    sys.stdout.flush()


def print_info(lines):
    """Prints informations messages, enhanced graphical aspects."""
    print Fore.YELLOW + "-------------------------  Informations  -------------------------\r"
    for line in lines:
        print Fore.YELLOW + "| " + line + '\r'
    print Fore.YELLOW + "------------------------------------------------------------------\r" + Fore.WHITE

def print_error(lines):
    """Prints errors messages, enhanced graphical aspects."""
    print Fore.RED + "----------------------------  Error  -----------------------------\r"
    for line in lines:
        print Fore.RED + "| " + line + '\r'
    print Fore.RED + "------------------------------------------------------------------\r" + Fore.WHITE

def get_colored_header():
    return Fore.GREEN + get_header() + Fore.RESET
