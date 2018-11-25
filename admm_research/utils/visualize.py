import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from visdom import Visdom
import copy, os, shutil
import matplotlib.pyplot as plt
from admm_research.utils import dice_loss
# plt.switch_backend('agg')

class Dashboard:

    def __init__(self, server='http://localhost', port=8097, env='default'):
        self.vis = Visdom(port=port, server=server, env=env)
        self.index = {}
        self.log_text = ''

    def loss(self, losses, title):
        x = np.arange(1, len(losses) + 1, 1)

        self.vis.line(losses, x, env='loss', opts=dict(title=title))

    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        self.vis.image(image.astype(np.float), env=self.vis.env, opts=dict(title=title))

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1


class Writter_tf(SummaryWriter):

    def __init__(self, writer_name: str, torchnet, num_img=20, random_seed=1, device='cpu') -> None:
        super().__init__(log_dir=writer_name)
        assert isinstance(num_img, int)
        self.writer_name = writer_name
        self.torchnet = torchnet
        self.random_seed = random_seed
        self.num_img = num_img
        self.device = device

    def cleanup(self, src='runs', des='archive'):
        self.export_scalars_to_json(os.path.join(self.writer_name, 'json.json'))
        self.close()
        writerbasename = os.path.basename(self.writer_name)
        shutil.move(os.path.join(src, writerbasename), os.path.join(des, writerbasename))

    def customized_add_image(self, img, gt, weak_gt, path, epoch):
        assert img.size(0) == 1

        fig = plt.figure()
        plt.imshow(img.data.cpu().squeeze().numpy(), cmap='gray')
        plt.contour(gt.data.cpu().squeeze().numpy(), levels=[0.5], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        plt.contour(weak_gt.data.cpu().squeeze().numpy(), levels=[0.5], colors="yellow", alpha=0.2, linewidth=0.001,
                    label='GT')
        pred = self.torchnet(img).max(1)[1]
        [_, dice] = dice_loss(pred, gt)
        plt.contour(pred.data.cpu().squeeze().numpy(), levels=[0.5], level=[0],
                    colors="red", alpha=0.2, linewidth=0.001, label='CNN')
        plt.title('dice:%.3f' % dice)
        plt.axis('off')
        self.add_figure(path, fig, global_step=epoch)

    def add_images(self, dataloader, epoch, device='cpu'):
        dataset_name = dataloader.dataset.name
        np.random.seed(self.random_seed)
        dataloader_ = copy.deepcopy(dataloader)
        np.random.seed(self.random_seed)
        selected_indxs = np.random.permutation(dataloader.dataset.__len__())[:self.num_img]
        selected_imgs = [dataloader.dataset.imgs[indx] for indx in selected_indxs]
        dataloader_.dataset.imgs = selected_imgs
        for i, (img, gt, weak_gt, path) in enumerate(dataloader_):
            if gt.sum() == 0 or weak_gt.sum() == 0:
                continue
            img, gt, weak_gt = img.to(device), gt.to(device), weak_gt.to(device)
            self.customized_add_image(img, gt, weak_gt, os.path.join(dataset_name, os.path.basename(path[0])), epoch)
