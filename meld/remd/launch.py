import logging
from meld import vault
from meld.system import get_runner


logger = logging.getLogger(__name__)


def launch(debug=False):
    store = vault.DataStore.load_data_store()

    communicator = store.load_communicator()
    communicator.initialize()

    if communicator.is_master():
        level = logging.DEBUG if debug else logging.INFO
        format = '%(asctime)s  %(name)s: %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        logging.basicConfig(filename='remd.log', level=level, format=format,
                            datefmt=datefmt)
        logger.info('Launching replica exchange')
    else:
        if debug:
            format = '%(asctime)s  %(name)s: %(message)s'
            datefmt = '%Y-%m-%d %H:%M:%S'
            logging.basicConfig(level=logging.DEBUG, format=format,
                                datefmt=datefmt)
            logger.info('Launching replica exchange')

    system = store.load_system()
    options = store.load_run_options()

    system_runner = get_runner(system, options, communicator)

    if communicator.is_master():
        store.initialize(mode='a')
        remd_runner = store.load_remd_runner()
        remd_runner.run(communicator, system_runner, store)
    else:
        remd_runner = store.load_remd_runner().to_slave()
        remd_runner.run(communicator, system_runner)
