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


def set_logger(params):
    f_next, extension = os.path.splitext(params.get('data', 'data_file'))
    log_file          = f_next + '.log'
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', 
        filename=log_file,
        level=logging.DEBUG, 
        datefmt='%m/%d/%Y %I:%M:%S %p')

def write_to_logger(params, to_write, level='info'):
    set_logger(params)
    for line in to_write:
        if level == 'info':
            logging.info(line)
        elif level in ['debug', 'default']:
            logging.debug(line)
        elif level == 'warning':
            logging.warning(line)


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
    print Fore.YELLOW + "------------------------------------------------------------------\r"

def print_error(lines):
    """Prints errors messages, enhanced graphical aspects."""
    print Fore.RED + "----------------------------  Error  -----------------------------\r"
    for line in lines:
        print Fore.RED + "| " + line + '\r'
    print Fore.RED + "------------------------------------------------------------------\r"