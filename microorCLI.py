import click
import logging

import microor
from pudb import set_trace as st

single_process = False

@click.group()
@click.version_option(0.1)
@click.argument('template')
@click.argument('image-folder')
@click.option('-v', '--verbose', count=True)
@click.option('--single-process', 'run_single_process', is_flag=True, help='If set will run in a single process')
def microorCLI(template, image_folder, verbose, run_single_process):
    """
    Automatic Object Recognition and quantification of micro array images
    """
    if verbose >= 3:
        logconsolelevel = logging.DEBUG
    elif verbose >= 2:
        logconsolelevel = logging.INFO
    else:
        logconsolelevel = logging.INFO
 
    #Setup logging to file
    logging.basicConfig(level=logging.DEBUG, filename='last_log.txt', filemode="w")

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
    

    # TODO: Load configfy configuration file based on template
    slide_experiment(image_folder, single_process)


if __name__ == '__main__':
    microorCLI()
