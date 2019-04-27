import warnings
from pprint import pprint

import yaml
from pathlib2 import Path

from admm_research.dataset import loader_interface
from admm_research.loss import get_loss_fn
from admm_research.method import get_method_class
from admm_research.models import Segmentator
from admm_research.trainer.trainer_refactor import ADMM_Trainer
from admm_research.utils import yaml_parser, dict_merge

warnings.filterwarnings('ignore')


def main(config: dict):
    model = Segmentator(config['Arch'], config['Optim'], config['Scheduler'])

    train_loader, val_loader = loader_interface(config['Dataset'], config['Dataloader'])

    admmmethod = get_method_class(config['ADMM_Method']['name'])(model=model,
                                                                 **{k: v for k, v in config['ADMM_Method'].items() if
                                                                    k != 'name'})
    trainer = ADMM_Trainer(
        ADMM_method=admmmethod,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=get_loss_fn(config['Loss']['name']),
        whole_config_dict=config,
        **config['Trainer']
    )
    whole_results = trainer.start_training()
    summary_output = trainer.summary()
    return summary_output, whole_results


if __name__ == '__main__':
    parser_args = yaml_parser()
    print('->>Input args:')
    pprint(parser_args)
    with open('config_3D_RV.yaml') as f:
        config = yaml.load(f, )
    config = dict_merge(config, parser_args, True)

    # overwrite the checkpoint config
    if config.get('Trainer', {}).get('checkpoint') is not None:
        try:
            with open(f"{Path(config['Trainer']['checkpoint']) / 'config_ACDC.yaml'}") as f:
                config = yaml.load(f, )
            config = dict_merge(config, parser_args, True)
        except (KeyError, FileNotFoundError) as e:
            print(f'Load saved config file failed with error: {e}, using initial config + argparser.')

    print('>>Merged args:')
    pprint(config)
    whole_result, best_results = main(config)
    print(best_results)
