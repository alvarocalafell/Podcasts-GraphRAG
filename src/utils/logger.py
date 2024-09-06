import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Setup loggers
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(logs_dir, exist_ok=True)

main_logger = setup_logger('main_logger', os.path.join(logs_dir, 'main.log'))
graph_logger = setup_logger('graph_logger', os.path.join(logs_dir, 'graph.log'))
model_logger = setup_logger('model_logger', os.path.join(logs_dir, 'model.log'))