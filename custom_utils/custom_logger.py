from logging import handlers
import logging
import colorlog
import os
import sys

class Logger:
    def __init__(self, mode, filePath=None):
        self.className = "Logger"
        self.mode = mode
        self.filePath = filePath
        
        if self.filePath is None:
            if self.mode == 'train':
                self.filePath = "./logs/checkpoints/"

            elif self.mode == 'test':
                self.filePath = "./logs/inference/"
        else:
            self.filePath = filePath
            
        if not os.path.exists(self.filePath):
            os.makedirs(self.filePath)

        
    def initLogger(self):
        __logger = logging.getLogger("Logger")

        streamFormatter = colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)-8s]%(reset)s <%(name)s>: %(module)s:%(lineno)d:  %(bg_blue)s%(message)s"
        )
        fileFormatter = logging.Formatter(
            #"%(asctime)s [%(levelname)-8s] <%(name)s>: %(module)s:%(lineno)d: %(message)s"
            "%(message)s"
        )
        streamHandler = colorlog.StreamHandler(sys.stdout)
    
        if self.mode == 'train':
            save_path = os.path.abspath(f"{self.filePath}checkpoint_path.yaml")

        elif self.mode == 'test':
            save_path = os.path.abspath(f"{self.filePath}test_result.yaml")

        fileHandler = handlers.TimedRotatingFileHandler(
            save_path,
            #os.path.abspath(f"checkpoint_path.yaml"),
            when="midnight",
            interval=1,
            backupCount=14,
            encoding="utf-8",
        )
        streamHandler.setFormatter(streamFormatter)
        fileHandler.setFormatter(fileFormatter)

        __logger.addHandler(streamHandler)
        __logger.addHandler(fileHandler)
        __logger.setLevel(logging.DEBUG)

        return __logger