import yaml
from pprint import pprint

from admm_research.models import Segmentator
from admm_research.dataset import loader_interface

with open('config.yaml') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)
pprint(config)

model = Segmentator(config['Arch'],config['Optim'],config['Scheduler'])

train_loader, val_loader = loader_interface(config['Dataset'],config['Dataloader'])
print(iter(train_loader).__next__())