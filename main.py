import yaml
from pprint import pprint
from torch import nn
from admm_research.models import Segmentator
from admm_research.dataset import loader_interface
from admm_research.trainer.trainer_refactor import ADMM_Trainer
from admm_research.method.ADMM_refactor import AdmmSize
from admm_research.loss import get_loss_fn

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
pprint(config)

model = Segmentator(config['Arch'], config['Optim'], config['Scheduler'])

train_loader, val_loader = loader_interface(config['Dataset'], config['Dataloader'])
# print(iter(train_loader).__next__())
admmmethod = AdmmSize(model=model,OptimInnerLoopNum=1,device='cuda')

trainer = ADMM_Trainer(
    ADMM_method=admmmethod,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    criterion=get_loss_fn(config['Loss']['name'])
)
trainer.start_training()
