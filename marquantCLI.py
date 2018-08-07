import click
import logging
import configfy

from pudb import set_trace as st

import marquant

single_process = False

@click.command()
@click.version_option(0.2)
@click.argument('image-folder')
@click.argument('config')
@click.option('-v', '--verbose', count=True)
@click.option('--single-process', 'run_single_process', is_flag=True, help='If set will run in a single process')
def marquantCLI(image_folder, config, verbose, run_single_process):
    """
    Automatic Object Recognition and quantification of micro array images
    """
    if verbose >= 2:
        logconsolelevel = logging.DEBUG
    elif verbose >= 1:
        logconsolelevel = logging.INFO
    else:
        logconsolelevel = logging.WARN
 
    #Setup logging to file
    logging.basicConfig(level=logging.DEBUG, filename='last_log.txt', filemode='w')

    #Add logging to the console
    console = logging.StreamHandler()
    console.setLevel(logconsolelevel)
    logging.getLogger('').addHandler(console)

    if run_single_process:
        logging.info('Running in single process mode!')
        global single_process
        single_process = True
    else:
        logging.info('Running in multiprocessing process mode!')
    
    current_config = configfy.set_active_config_file(config)
    if current_config is None:
        logging.error('Cannot load config file. Will abort!')
        exit(1)

    # TODO: Load configfy configuration file based on template
    marquant.slide_experiment(image_folder, single_process)


if __name__ == '__main__':
    marquantCLI()
