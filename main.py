import argparse
import os
from pathlib import Path
import math
import colossalai
import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CosineAnnealingLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits

model = LeNet5(n_classes=10)
model.to("cuda")



def lr_range_test(optimizer, args):
    config = {'BATCH_SIZE': 128, 'NUM_EPOCHS': 5}
    colossalai.launch(config=config, rank=0, world_size=1, host='127.0.0.1', port=1234)
    logger = get_dist_logger()
    # build criterion
    criterion = torch.nn.CrossEntropyLoss()

    train_dataset = MNIST(
        root=Path('./tmp/'),
        download=True,
        transform=transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()])
    )
    test_dataset = MNIST(
        root=Path('./tmp/'),
        train=False,
        transform=transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()])
    )
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      num_workers=1,
                                      pin_memory=True,
                                      )

    test_dataloader = get_dataloader(dataset=test_dataset,
                                     add_sampler=False,
                                     batch_size=gpc.config.BATCH_SIZE,
                                     num_workers=1,
                                     pin_memory=True,
                                     )
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                         optimizer,
                                                                         criterion,
                                                                         train_dataloader,
                                                                         test_dataloader,
                                                                         )
    # build a timer to measure time
    timer = MultiTimer()

    def lrs(batch):
        low = math.log2(1e-5)
        high = math.log2(10)
        return 2 ** (low + (high - low) * batch / len(train_dataloader) / gpc.config.NUM_EPOCHS)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrs)
    # create a trainer object
    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger
    )

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        # hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),

        # you can uncomment these lines if you wish to use them
        hooks.TensorboardHook(log_dir='./tb_logs_ADAM_test', ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_dataloader=test_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )

def training(optimizer, args):
    config = {'BATCH_SIZE': 128, 'NUM_EPOCHS': 30}
    colossalai.launch(config=config, rank=0, world_size=1, host='127.0.0.1', port=1234)
    logger = get_dist_logger()
    # build criterion
    criterion = torch.nn.CrossEntropyLoss()
    train_dataset = MNIST(
        root=Path('./tmp/'),
        download=True,
        transform=transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()])
    )
    test_dataset = MNIST(
        root=Path('./tmp/'),
        train=False,
        transform=transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()])
    )
    train_dataloader = get_dataloader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=gpc.config.BATCH_SIZE,
                                      num_workers=1,
                                      pin_memory=True,
                                      )

    test_dataloader = get_dataloader(dataset=test_dataset,
                                     add_sampler=False,
                                     batch_size=gpc.config.BATCH_SIZE,
                                     num_workers=1,
                                     pin_memory=True,
                                     )
    if args.scheduler == "MultiStepLR":
        if args.MS_int == "uniform":
            mile_stone = [3000, 6000, 9000, 12000]
        else:
            mile_stone = [500, 1500, 4500, 13500]
        log_name = './tb_logs_{}_MUltiStepLR_{}_{}_train'.format(args.optimizer, args.MS_int, args.LR)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stone, gamma=0.2)
    elif args.scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Cos_int)
        log_name = './tb_logs_{}_CosineAnnealingLR_{}_{}_train'.format(args.optimizer, args.Cos_int, args.LR)
    else:
        print("unknown scheduler")
        exit()
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                         optimizer,
                                                                         criterion,
                                                                         train_dataloader,
                                                                         test_dataloader,
                                                                         )
    # build a timer to measure time
    timer = MultiTimer()

    # create a trainer object
    trainer = Trainer(
        engine=engine,
        timer=timer,
        logger=logger
    )

    # define the hooks to attach to the trainer
    hook_list = [
        hooks.LossHook(),
        hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
        # hooks.AccuracyHook(accuracy_func=Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogMemoryByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer, logger),

        # you can uncomment these lines if you wish to use them
        hooks.TensorboardHook(log_dir=log_name, ranks=[0]),
        # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
    ]

    # start training
    trainer.fit(
        train_dataloader=train_dataloader,
        epochs=gpc.config.NUM_EPOCHS,
        test_dataloader=test_dataloader,
        test_interval=1,
        hooks=hook_list,
        display_progress=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', default="test", type=str,
                        help='whether to perform LR range test or training, input is test or train')
    parser.add_argument('--optimizer', default="SGD", type=str,
                        help='optimizer')
    parser.add_argument('--scheduler', default="MultiStepLR", type=str,
                        help='MultiStepLR or CosineAnnealingLR')
    parser.add_argument('--MS_int', default="uniform", type=str,
                        help='which type of interval for MultiStep')
    parser.add_argument('--Cos_int', default=10, type=int,
                        help='T_MAX for CosineAnnealingLR')
    parser.add_argument('--LR', default=0.1, type=float,
                        help='training learning rate')

    args = parser.parse_args()
    if args.process == "test":
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == "ADAM":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        else:
            print("unknown oprimizer")
            exit()
        lr_range_test(optimizer, args)
    else:
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.LR, momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == "ADAM":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=5e-4)
        else:
            print("unknown optimizer")
            exit()
        training(optimizer, args)
