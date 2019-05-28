from unittest import TestCase
from .ACDC_helper import ACDC_dataloader
from . import loader_interface


class TestACDC_dataloader(TestCase):
    def setUp(self):
        self.dataset_config = {'dataset_name': 'cardiac',
                               'use_data_aug': True}
        self.data_loader_config = {'num_workers': 3,
                                   'batch_size': 4}

    def test_ACDC_dataloader(self):
        train_loader, val_loader = ACDC_dataloader(self.dataset_config, self.data_loader_config)
        print(iter(train_loader).__next__().__len__())
        loader_interface(self.dataset_config, self.data_loader_config)
