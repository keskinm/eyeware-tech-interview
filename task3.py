import torch
from torch import nn
from torch.utils.data import DataLoader
from models.seven_conv import seven_conv
from models.two_conv import two_conv
from numpy import linalg as LA
from dataset.simulated_dataset import SimulatedDataset
import os
import argparse

class TopLeftCornerModelEvaluator:
    def __init__(self, train_epoch, lr, train_batch_size, test_model,
                 model_type, seed, save_dir, resume_model, optimizer,
                 dump_metrics_frequency, num_threads, simulated_dataset_size):
        self.train_epoch = train_epoch
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.train_batch_size = train_batch_size
        self.model_type = model_type
        if seed is not None:
            torch.manual_seed(seed)
        self.save_dir = save_dir
        self.loss_plots_dir = os.path.join(save_dir, 'losses_plots')
        self.save_model_dir_path = os.path.join(save_dir, 'models')
        self.metrics_dir_path = os.path.join(self.save_dir, 'metrics')

        os.makedirs(self.loss_plots_dir, exist_ok=True)
        os.makedirs(self.save_model_dir_path, exist_ok=True)
        os.makedirs(self.metrics_dir_path, exist_ok=True)

        self.simulated_dataset_size = simulated_dataset_size
        self.num_threads = num_threads
        self.train_set_loader, self.val_set_loader, self.test_set_loader = self.prepare_data(
        )

        self.optimizer = optimizer
        self.dump_metrics_frequency = dump_metrics_frequency

    def prepare_data(self):
        train_dataset_size = round(0.7 * self.simulated_dataset_size)
        val_dataset_size = round(0.15 * self.simulated_dataset_size)
        test_dataset_size = round(0.15 * self.simulated_dataset_size)

        if self.train_batch_size >= train_dataset_size:
            self.train_batch_size = train_dataset_size // 3

        train_dataset = SimulatedDataset(work_dir='./data/task3/images', len_data=train_dataset_size)
        val_dataset = SimulatedDataset(work_dir='./data/task3/images', len_data=val_dataset_size)
        test_dataset = SimulatedDataset(work_dir='./data/task3/images', len_data=test_dataset_size)

        train_set_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, num_workers=self.num_threads)
        val_set_loader = DataLoader(val_dataset, batch_size=self.train_batch_size, num_workers=self.num_threads)
        test_set_loader = DataLoader(test_dataset, batch_size=self.train_batch_size, num_workers=self.num_threads)

        return train_set_loader, val_set_loader, test_set_loader

    def run(self):
        seven_conv_model = seven_conv()
        two_conv_model = two_conv()
        if self.model_type == 'seven_conv':
            model = seven_conv_model
        else:
            model = two_conv_model
        model = model.double()
        criterion, optimizer = self.init_optimizer(model)
        self.train(model=model, criterion=criterion, optimizer=optimizer, device=self.device)

    def train(self, model, criterion, optimizer, device):
        losses = []
        epoch_n = self.train_epoch
        model = model.train()
        batch_id = None

        for epoch in range(1, epoch_n + 1):
            for batch_id, (image, label) in enumerate(self.train_set_loader):
                label, image = label.to(device), image.to(device)
                output = model(image)
                loss = criterion(output, label.double())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (batch_id != 0) and (batch_id % self.dump_metrics_frequency == 0):
                    mse = self.compute_mse(model, self.val_set_loader, device=self.device)
                    self.dump_metrics_and_save_model(mse=mse,  epoch=epoch, losses=losses, model=model,
                                                     model_name=self.model_type, optimizer=optimizer, batch_id=batch_id)
                    self.save_model(epoch=epoch, losses=losses, model=model, optimizer=optimizer, model_name=self.model_type)

            print("epoch:", epoch)
            mse = self.compute_mse(model, self.val_set_loader, device=self.device)
            self.dump_metrics_and_save_model(mse=mse, epoch=epoch, losses=losses, model=model,
                                             model_name=self.model_type, optimizer=optimizer, batch_id=batch_id)
            self.save_model(epoch=epoch, losses=losses, model=model, optimizer=optimizer, model_name=self.model_type)

    def dump_metrics_and_save_model(self, mse, epoch, losses, model, model_name, optimizer,
                                    batch_id):
        self.dump_mse(mse, model_name, epoch, batch_id)
        self.save_model(epoch=epoch, losses=losses, model=model, optimizer=optimizer, model_name=self.model_type)

    def dump_mse(self, mse, model_name, epoch, batch_idx):
        metrics_file_path = os.path.join(self.metrics_dir_path,
                                         '{}.txt'.format(model_name))
        with open(metrics_file_path, "a") as opened_metrics_file:
            opened_metrics_file.write(
                "epoch:{epoch} batch_idx:{batch_idx} mse:{mse}\n"
                .format(epoch=epoch,
                        batch_idx=batch_idx,
                        mse=mse))

    def save_model(self, epoch, losses, model, optimizer, model_name):
        save_model_file_path = os.path.join(self.save_model_dir_path,
                                            '{}.pth'.format(model_name))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses
            }, save_model_file_path)

    def init_optimizer(self, model):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                model.parameters()),
                                         lr=self.lr)

        else:
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                               model.parameters()),
                                        lr=self.lr,
                                        momentum=0.9)

        criterion = nn.MSELoss()
        return criterion, optimizer

    def compute_mse(self, model, data_set, device):
        model = model.eval()
        with torch.no_grad():
            mse = 0
            total = 0
            for batch_id, (image, label) in enumerate(data_set):
                image = image.to(device)
                label = label.to(device)
                outputs = model(image).to(device)
                se = LA.norm(outputs-label.double())
                # print(se)
                total += label.shape[0]/4*2
                mse += se
        print(mse)
        return mse


def main():
    parser = argparse.ArgumentParser(prog='Simulated data wise left corner detection model evaluator')

    parser.add_argument('--model-type',
                        choices=['two_conv', 'seven_conv'],
                        default='two_conv',
                        help='')

    parser.add_argument('-t',
                        '--test-model',
                        nargs='+',
                        help='model path and model name',
                        default=None)

    parser.add_argument('-r',
                        '--resume-model',
                        nargs='+',
                        help='model path and model name',
                        default=None)

    parser.add_argument('--train-batch-size', default=10, help='')

    parser.add_argument('--lr', default=0.005, type=float, help='')

    parser.add_argument('--train-epoch', default=20, type=int, help='')

    parser.add_argument('--seed', default=42, help='')

    parser.add_argument('--save-dir', default='./data/task3', help='')

    parser.add_argument('--optimizer',
                        choices=['adam', 'sgd'],
                        default='adam',
                        help='')

    parser.add_argument('--dump-metrics-frequency',
                        metavar='Batch_n',
                        default='10',
                        type=int,
                        help='Dump metrics every Batch_n batches')

    parser.add_argument(
        '--num-threads',
        default='0',
        type=int,
        help='Number of CPU to use for processing mini batches')

    parser.add_argument(
        '--simulated-dataset-size',
        default='80',
        type=int,
        help='Number of samples (train+val+test)')

    args = parser.parse_args()
    args = vars(args)
    evaluator = TopLeftCornerModelEvaluator(**args)
    evaluator.run()


if __name__ == "__main__":
    main()