import yaml
from pprint import pprint
from admm_research.models import Segmentator
from admm_research.dataset import loader_interface
from admm_research.trainer.trainer_refactor import ADMM_Trainer
from admm_research.method import get_method_class
from admm_research.loss import get_loss_fn
import warnings
warnings.filterwarnings('ignore')

with open('config.yaml') as f:
    config = yaml.load(f, )
pprint(config)

model = Segmentator(config['Arch'], config['Optim'], config['Scheduler'])

train_loader, val_loader = loader_interface(config['Dataset'], config['Dataloader'])

admmmethod = get_method_class(config['ADMM_Method']['name'])(model=model,**{k:v for k,v in config['ADMM_Method'].items() if k !='name'})

trainer = ADMM_Trainer(
    ADMM_method=admmmethod,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    criterion=get_loss_fn(config['Loss']['name']),
    whole_config_dict=config,
    **config['Trainer']
)
trainer.start_training()
