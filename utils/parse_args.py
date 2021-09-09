import argparse
import time

def str_to_bool(in_str) :
    
    if in_str == 'True' : return True
    elif in_str == 'False' : return False

    assert in_str in ['True', 'False'], 'argparser gets wrong boolean type.'

def Parsing_Args(args) :
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="python run.py mode")

    parser.add_argument(
        '--seed', type = int, default = 42,
        help = 'seed number that fixes randomness'
    )
    parser.add_argument(
        '--data', type = str, default = 'mini_imagenet',
        help = 'select dataset for training'
    )
    parser.add_argument(
        '--server', type = str_to_bool, default = True,
        help = 'select whether server or local environment'
    )
    parser.add_argument(
        '--batch_size', type = int, default=32,
        help = 'set batch size'
    )
    parser.add_argument(
        '--epoch', type = int, default = 60,
        help = 'set training epoch'
    )
    parser.add_argument(
        '--device', type = int, default = 0,
        help = 'select which GPU to use'
    )
    parser.add_argument(
        '--baseline', type = str_to_bool, default = False,
        help = 'select whether baseline or target model'
    )
    parser.add_argument(
        '--file', type = str, default = 'model_parameter',
        help = 'name your saved model parameters'
    )
    now = time.localtime()
    parser.add_argument(
        '--dst', type = str, default = '{:02d}_{:02d}_{:02d}-{:02d}_{:02d}_{:02d}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec),
        help = 'folder that save model parameter and log.txt'
    )
    parser.add_argument(
        '--model', type = str, default = 'resnet50',
        help = 'select a model you want to train'
    )

    return vars(parser.parse_known_args(args)[0])