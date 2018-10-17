import os

project_root_path = os.path.dirname(os.getcwd())
if not os.path.isdir(os.path.join(project_root_path,'dataset')):
    os.mkdir(os.path.join(project_root_path,'dataset'))
if not os.path.isdir(os.path.join(project_root_path, 'semi_pretrain_checkpoint')):
    os.mkdir(os.path.join(project_root_path, 'semi_pretrain_checkpoint'))
if not os.path.isdir(os.path.join(project_root_path, 'results')):
    os.mkdir(os.path.join(project_root_path, 'results'))
cmd_ = '"cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"'
os.system('/bin/bash -c ' + cmd_)
cmd = '"kaggle datasets download -d jizongpeng/acdc2dall --force -p %s"'%os.path.join(project_root_path,'dataset')
os.system('/bin/bash -c '+ cmd)
