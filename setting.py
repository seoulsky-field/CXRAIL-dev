import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

from ray.tune import loguniform, choice, uniform, randint

from data_loader.data_loader import CXRDataset

class LoadExperimentSetting:
    def __init__(
        self,
        hydra_cfg,
        model,
       ):
        self.search_space = {}
        self.hydra_cfg = hydra_cfg
        self.loss_f = instantiate(hydra_cfg.loss)
        try:
            self.optimizer = instantiate(
                hydra_cfg.optimizer,
                params=model.parameters()
            )
        except:
            self.optimizer = instantiate(
                hydra_cfg.optimizer,
                model=model,
                loss_fn=self.loss_f
            )
        self.train_dataset = CXRDataset("train", **hydra_cfg.Dataset, )
        self.val_dataset = CXRDataset("valid", **hydra_cfg.Dataset,)

        self.train_loader = instantiate(hydra_cfg.Dataloader.train, dataset=self.train_dataset)
        self.val_loader = instantiate(hydra_cfg.Dataloader.train, dataset=self.val_dataset)
        
        self.tune_list = hydra_cfg.hparams_search.get('tune_list', {})

    # def setup(self, ):
        #model-> loss_f -> data -> optimizer


    def detect_list(self, target):
        if self.tune_list.get(target, None) == True:
            return True
        else:
            return False

    def set_dataloader(self):
        augmentation_search_space = {}
        if self.detect_list('augmentation'):
            augmentation_search_space['num_ops'] = randint(2, 14) #ra_num_ops,
            augmentation_search_space['magnitude'] = randint(5, 20) #ra_magnitude,
            self.search_space += augmentation_search_space

            self.train_dataset.augmentation_search_space = augmentation_search_space
            self.train_loader.dataset = self.train_dataset

        if self.detect_list('batch_size'):
            self.search_space['batch_size'] = choice[128, 256]
            self.train_loader.batch_size = self.search_space['batch_size']
            self.val_loader.batch_size = self.search_space['batch_size']

        print("-------load dataloader--------")
        return self.train_loader, self.val_loader


    def set_optimizer(self, mode='soft'):

        if self.detect_list("optimizer"):
            # 공통 - soft
            self.search_space['lr'] = loguniform(optimizer.lr*0.01, optimizer.lr*100)   
            self.search_space['weight_decay']= loguniform(optimizer.weight_decay*0.01, optimizer.weight_decay*100)

            if mode == "hard":
                if 'torch':
                    # betas_1 = loguniform(0.00001, 0.1) # 0.9
                    # betas_2 = loguniform(0.99)
                    # betas = (betas_1, betas_2)
                    self.search_space['eps'] = loguniform(optimizer.eps*0.01, optimizer.eps*100)   #1e-8

                elif 'RMSprop':
                    # momentum: 0   # momentum factor (default: 0)
                    self.search_space['alpha'] = loguniform(optimizer.alpha*0.01, optimizer.alpha*100) #0.99  
                    self.search_space['eps'] = loguniform(optimizer.eps*0.01, optimizer.eps*100)
                    #centered: True
                elif 'PESG':
                    self.search_space['margin'] = loguniform(optimizer.margin*0.01, optimizer.margin*100)#: 1.0
                    #epoch_decay: 2e-3
        
            for keys, values in search_space.items():
                self.optimizer[keys] = values
        
        print("-------load optimizer--------")
        return self.optimizer

    def set_criterion(self):
        if self.detect_list('loss'):
            if 'AsymmetricLoss' in hydra_cfg.loss.__target__:
                self.search_space['gamma_neg'] = uniform(1, 5)
                #gamma_pos = uniform(0.05, 0.25)
                self.search_space['clip'] = uniform(0.05, 0.25)
                self.search_space['disable_torch_grad_focal_loss'] = choice['True', 'False']

            for keys, values in search_space.items():
                self.loss_f[keys] = values
        print("-------load loss_f--------")
        return self.loss_f

    
    def return_search_space(self):
        if self.search_space != {}:
            # for keys, values in self.search_space.items():
            #     print("")
            print(f"Search Space : {self.search_space}")
            return self.search_space
            
        
        else:
            print("No Search space ")
        

        # # torch otim
        # ## adam
        # lr = loguniform(0.00001, 0.1)
        # betas_1 = loguniform(0.00001, 0.1) # 0.9
        # betas_2 = loguniform(0.99)
        # betas = (betas_1, betas_2)
        # eps: loguniform()#1e-8
        # weight_decay: loguniform()#1e-8

        # ##  RMSproptorch.optim.RMSprop
        # lr: loguniform(optimizer.lr*0.01, optimizer.lr*100)    # (default: 1e-2)
        # momentum: 0   # momentum factor (default: 0)
        # alpha: 0.99   # smoothing constant (default: 0.99)
        # eps: loguniform(optimizer.eps*0.01, optimizer.eps*100)#1e-8
        # centered: True
        # weight_decay: loguniform(optimizer.weight_decay*0.01, optimizer.weight_decay*100)

        # # pesg
        # margin: 1.0
        # epoch_decay: 2e-3
        # weight_decay:  1e-5

            # # search space
    # lr: float = config.get("lr", hydra_cfg["lr"])
    # weight_decay: float = config.get("weight_decay", hydra_cfg["lr"])
    # batch_size: int = config.get("batch_size", hydra_cfg["batch_size"])
    # asl_gamma_neg: int = config.get("asl_gamma_neg", hydra_cfg["asl_gamma_neg"])
    # asl_ps_factor: float = config.get("asl_ps_factor", hydra_cfg["asl_ps_factor"])
    # ra_num_ops: int = config.get("ra_num_ops", hydra_cfg["ra_num_ops"])
    # ra_magnitude: int = config.get("ra_magnitude", hydra_cfg["ra_magnitude"])
    # ra_params = {  # binding random augment parameters
    #     "num_ops": ra_num_ops,
    #     "magnitude": ra_magnitude,
    # }
# hydra

#Training can be done with either a Function API (session.report) or Class API (tune.Trainable).

