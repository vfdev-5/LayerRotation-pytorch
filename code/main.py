
# Train / Test CNN on CIFAR10
# Optionally scripts uses Layca algorithm from 
# "Layer rotation: a surprisingly powerful indicator of 
# generalization in deep networks?"
# https://arxiv.org/pdf/1806.01603v2.pdf

import argparse
import traceback
from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.utils import convert_tensor

from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler as tbOutputHandler, \
    OptimizerParamsHandler as tbOptimizerParamsHandler

from ignite.contrib.handlers import PiecewiseLinear

import mlflow

from utils import set_seed, get_train_test_loaders, get_model
from handlers.layer_rotation import LayerRotationStatsHandler
from layca_optims.sgd import LaycaSGD


def run(output_path, config):

    device = "cuda"
    batch_size = config['batch_size']

    train_loader, test_loader = get_train_test_loaders(dataset_name=config['dataset'],
                                                       path=config['data_path'],
                                                       batch_size=batch_size,
                                                       num_workers=config['num_workers'])

    model = get_model(config['model'])
    model = model.to(device)
    
    optim_fn = optim.SGD
    if config['with_layca']:
        optim_fn = LaycaSGD

    optimizer = optim_fn(model.parameters(), lr=0.0,
                         momentum=config['momentum'],
                         weight_decay=config['weight_decay'],
                         nesterov=True)
    criterion = nn.CrossEntropyLoss()

    le = len(train_loader)
    milestones_values = [(le * m, v) for m, v in config['lr_milestones_values']]
    scheduler = PiecewiseLinear(optimizer, "lr", milestones_values=milestones_values)

    def _prepare_batch(batch, device, non_blocking):
        x, y = batch
        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking))

    def process_function(engine, batch):
                
        x, y = _prepare_batch(batch, device=device, non_blocking=True)
        
        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    trainer = Engine(process_function)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

    RunningAverage(output_transform=lambda x: x, epoch_bound=False).attach(trainer, 'batchloss')

    ProgressBar(persist=True, bar_format="").attach(trainer,
                                                    event_name=Events.EPOCH_STARTED,
                                                    closing_event_name=Events.COMPLETED)

    tb_logger = TensorboardLogger(log_dir=output_path)
    tb_logger.attach(trainer,
                     log_handler=tbOutputHandler(tag="train", metric_names=['batchloss', ]),
                     event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer,
                     log_handler=tbOptimizerParamsHandler(optimizer, param_name="lr"),
                     event_name=Events.ITERATION_STARTED)

    tb_logger.attach(trainer, 
                     log_handler=LayerRotationStatsHandler(model),
                     event_name=Events.EPOCH_STARTED)                     

    metrics = {
        "accuracy": Accuracy(),
    }

    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    
    def run_validation(engine, val_interval):
        if (engine.state.epoch - 1) % val_interval == 0:
            train_evaluator.run(train_loader)
            evaluator.run(test_loader)

    trainer.add_event_handler(Events.EPOCH_STARTED, run_validation, val_interval=2)
    trainer.add_event_handler(Events.COMPLETED, run_validation, val_interval=1)

    tb_logger.attach(train_evaluator,
                     log_handler=tbOutputHandler(tag="train",
                                                 metric_names=list(metrics.keys()),
                                                 another_engine=trainer),
                     event_name=Events.COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=tbOutputHandler(tag="test",
                                                 metric_names=list(metrics.keys()),
                                                 another_engine=trainer),
                     event_name=Events.COMPLETED)

    def mlflow_batch_metrics_logging(engine, tag):
        step = trainer.state.iteration
        for name, value in engine.state.metrics.items():
            mlflow.log_metric("{} {}".format(tag, name), value, step=step)

    def mlflow_val_metrics_logging(engine, tag):
        step = trainer.state.epoch
        for name in metrics.keys():
            value = engine.state.metrics[name]
            mlflow.log_metric("{} {}".format(tag, name), value, step=step)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, mlflow_batch_metrics_logging, "train")
    train_evaluator.add_event_handler(Events.COMPLETED, mlflow_val_metrics_logging, "train")
    evaluator.add_event_handler(Events.COMPLETED, mlflow_val_metrics_logging, "test")

    trainer.run(train_loader, max_epochs=config['num_epochs'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a CNN on a dataset")
    
    parser.add_argument('dataset', type=str, choices=['CIFAR10', 'CIFAR100'],
                        help="Training/Testing dataset")

    parser.add_argument('network', type=str, help="CNN to train")

    parser.add_argument('--params', type=str,
                        help='Override default configuration with parameters: '
                        'data_path=/path/to/dataset;batch_size=64;num_workers=12 ...')

    args = parser.parse_args()

    dataset_name = args.dataset    
    network_name = args.network    
    
    print("Train {} on {}".format(network_name, dataset_name))    
    print("- PyTorch version: {}".format(torch.__version__))
    print("- Ignite version: {}".format(ignite.__version__))
    
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    print("- CUDA version: {}".format(torch.version.cuda))

    batch_size = 512
    num_epochs = 24
    config = {
        "dataset": dataset_name,
        "data_path": ".",

        "model": network_name,

        "momentum": 0.9,
        "weight_decay": 1e-4,
        "batch_size": batch_size,
        "num_workers": 10,

        "num_epochs": num_epochs,

        "lr_milestones_values": [(0, 0.0), (5, 1.0), (num_epochs, 0.0)],
        
        "with_layca": False  # Apply Layca algorithm from the paper
    }

    # Override config:
    if args.params:
        for param in args.params.split(";"):
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    print("\n")
    print("Configuration:")
    for key, value in config.items():
        print("\t{}: {}".format(key, value))
    print("\n")

    mlflow.log_params(config)

    # dump all python files to reproduce the run
    mlflow.log_artifacts(Path(__file__).parent.as_posix())

    with tempfile.TemporaryDirectory() as tmpdirname:                        
        try:
            run(tmpdirname, config)
        except Exception as e:
            traceback.print_exc()
            mlflow.log_artifacts(tmpdirname)
            mlflow.log_param("run status", "FAILED")
            exit(1)

        mlflow.log_artifacts(tmpdirname)
        mlflow.log_param("run status", "OK")
