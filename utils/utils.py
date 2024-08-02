import os
import logging
import torch
import utils.distributed as dist

### obtain logger
class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        # create a logger
        self.__logger = logging.getLogger(logger_name)

        # set the log level
        self.__logger.setLevel(log_level)

        # create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        # create a handler to print on console
        console_handler = logging.StreamHandler()

        # define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


### make directory for experiment and return a logger
def mkExpDir(args):
    if dist.get_rank() == 0:
        if not (os.path.exists(args.save_dir)):
            os.makedirs(args.save_dir)
        
        # create sub save directory
        dirs = [name for name in os.listdir(args.save_dir) if os.path.isdir(os.path.join(args.save_dir, name))]
        if dirs:
            dirs = [int(dir) for dir in dirs if dir.isdigit()]
            max_dir = max(dirs) if len(dirs)>0 else 0
            args.save_dir = os.path.join(args.save_dir, "{:05d}".format(int(max_dir) + 1))
        else:
            args.save_dir = os.path.join(args.save_dir, "00000")
        os.makedirs(args.save_dir)
        
        # print save_dir
        dist.print0('save_dir: ' + args.save_dir)

        # create snapshot directory if needed
        if ((not args.eval) and (not args.test)):
            os.makedirs(os.path.join(args.save_dir, 'model'))

        # create result directory if needed
        if ((args.eval and args.eval_save_results) or args.test):
            os.makedirs(os.path.join(args.save_dir, 'results'))

        args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
        for k, v in vars(args).items():
            args_file.write(k.rjust(30, ' ') + '\t' + str(v) + '\n')

        _logger = Logger(log_file_name=os.path.join(args.save_dir, args.log_file_name),
                        logger_name=args.logger_name).get_log()
    else:
        _logger = logging.getLogger(args.logger_name)

    return _logger


### loading model 
def load_model(model, model_path, option=False):
    if os.path.exists(model_path):
        model_state_dict_save = {k.replace('module.', ''):v for k,v in torch.load(model_path).items()}
        model_state_dict = model.module.state_dict()
        model_state_dict.update(model_state_dict_save)
        model.module.load_state_dict(model_state_dict, strict=False)
    else:
        raise Exception('no model path! ' + str(model_path))
    

### loading UFormer model
def load_uformer(model, model_path):
    if os.path.exists(model_path):
        ckpt = torch.load(model_path)
        model_state_dict_save = {k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()}
        model_state_dict = model.module.state_dict()
        model_state_dict.update(model_state_dict_save)
        model.module.load_state_dict(model_state_dict)
    else:
        raise Exception('no model path!' + str(model_path))
