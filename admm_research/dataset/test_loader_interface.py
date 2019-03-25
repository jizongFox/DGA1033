from unittest import TestCase
from . import loader_interface

class TestLoader_interface(TestCase):
    def setUp(self):
        self.dataset_config = {'dataset_name': 'cardiac',
                               'use_data_aug': True}
        self.data_loader_config = {'num_workers': 3,
                                   'batch_size': 4}
    def test_loader_interface(self):
        train_loader, val_loader = loader_interface(self.dataset_config,self.data_loader_config)
        print(iter(train_loader).__next__())
        print(iter(val_loader).__next__())
