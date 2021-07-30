import logging
import logging.config
import math
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import data_normalization, fdutil, utils
from lib.AverageMeter import AverageMeter


class Trainer(object):
    def __init__(self, args):
        self.config = args

        self.save_dir = args.save_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.tboard_log_dir = args.tboard_log_dir
        self.pretrained_path = args.pretrained_path
        self.log_file = args.log_file

        fdutil.make_dir(self.save_dir)
        fdutil.make_dir(self.checkpoint_dir)
        self.path_model_best = os.path.join(self.checkpoint_dir, 'Model_best.pth')
        self.path_model_last = os.path.join(self.checkpoint_dir, 'Model_last.pth')

        self.writer = SummaryWriter(log_dir=self.tboard_log_dir)
        self.logger = utils.setup_logger('train_logger', level=logging.INFO, log_to_console=True,
                                         log_file=self.log_file)

        self.start_epoch = 0
        self.n_epochs = args.n_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.criterion = args.criterion

        self.evaluate_rate = args.evaluate_rate
        self.save_model_rate = args.save_model_rate

        # Frequency with which the training losses are averaged and logged
        self.freq_average_train_loss = args.freq_average_train_loss

        self.best_loss = math.inf
        self.index_best_loss = math.inf

        if self.pretrained_path is not None:
            self._load_pretrain(self.pretrained_path)
        else:
            self.logger.info('\nStart training from scratch.\n')
            self.model = self.model.to(self.device)

        self.loader = dict()
        self.loader['train'] = args.trainloader
        self.loader['val'] = args.valloader

        # Extract the batch size
        iterator = iter(self.loader['train'])
        batch = next(iterator)
        x, _, _ = self._extract_inputs_outputs_loss_masks(batch)
        self.batch_size = x.shape[0]

        # Extract the parameter settings for hyper-parameter logging
        self.hparams = dict()
        self.hparams['batch_size'] = self.batch_size
        self.hparams['lr_initial'] = self._get_lr()
        self.hparams['optimizer'] = self.optimizer.__class__.__name__

        if self.scheduler is not None:
            self.hparams['scheduler'] = self.scheduler.__class__.__name__

            if self.hparams['scheduler'] == 'ReduceLROnPlateau':
                self.hparams['patience'] = self.scheduler.patience
                self.hparams['step_size'] = -1  # hyper-parameters set to None do not get displayed on tensorboard

            elif self.hparams['scheduler'] == 'StepLR':
                self.hparams['patience'] = -1
                self.hparams['step_size'] = self.scheduler.step_size
        else:
            self.hparams['scheduler'] = 'None'
            self.hparams['patience'] = -1
            self.hparams['step_size'] = -1

    def _compute_denormalized_loss(self, y_pred, y, loss_mask, mean, std):
        y_pred_metric = data_normalization.denormalize_torch(y_pred, mean, std)
        y_metric = data_normalization.denormalize_torch(y, mean, std)

        # Select valid pixels only
        y_pred_metric[loss_mask == 0] = 0
        y_metric[loss_mask == 0] = 0

        # Correct for the number of valid pixels as the criterion averages the loss over all pixels
        # (including invalid pixels)
        loss = self.criterion(y_pred_metric, y_metric)
        loss = loss * loss_mask.numel() / loss_mask.sum()

        return loss

    @staticmethod
    def _extract_inputs_outputs_loss_masks(batch):
        x = batch['input']
        y = batch['target']
        loss_mask = batch['loss_mask']

        return x, y, loss_mask

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def _load_pretrain(self, resume):
        """
        Resumes training.

        :param resume:  str, path of the pretrained model weights
        """

        if os.path.isfile(resume):
            checkpoint = torch.load(resume)
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Push the model to GPU before loading the state of the optimizer
            self.model = self.model.to(self.device)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Extract the last training epoch
            self.start_epoch = checkpoint['epoch'] + 1
            self.n_epochs += self.start_epoch

            # Best validation loss so far
            self.best_loss = checkpoint['loss_val']
            self.index_best_loss = checkpoint['epoch']

            self.logger.info('\n\nRestoring the pretrained model from epoch {}.'.format(self.start_epoch))
            self.logger.info(f'Successfully load pretrained model from {resume}!\n')
            self.logger.info(f'Current best loss {self.best_loss}\n')
        else:
            raise ValueError(f"No checkpoint found at '{resume}.\n'")

    def _save_checkpoint(self, epoch, loss_train, loss_val, filepath):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_val': loss_val,
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state, filepath)

    def inference_one_batch(self, batch, phase):
        assert phase in ['train', 'val']

        if phase == 'train':
            self.model.train()

            x, y, loss_mask = self._extract_inputs_outputs_loss_masks(batch)
            x = x.to(self.device)
            y = y.to(self.device)
            loss_mask = loss_mask.to(self.device)

            # Forward pass
            y_pred = self.model(x)

            # Compute the denormalized loss
            mean = torch.flatten(batch['dsm_mean'])
            std = torch.flatten(batch['dsm_std'])
            loss = self._compute_denormalized_loss(y_pred, y, loss_mask, mean, std)

            # Backpropagation
            loss.backward()
        else:
            self.model.eval()

            with torch.no_grad():
                x, y, loss_mask = self._extract_inputs_outputs_loss_masks(batch)
                x = x.to(self.device)
                y = y.to(self.device)
                loss_mask = loss_mask.to(self.device)

                # Forward pass
                y_pred = self.model(x)

                # Compute the denormalized loss
                mean = torch.flatten(batch['dsm_mean'])
                std = torch.flatten(batch['dsm_std'])
                loss = self._compute_denormalized_loss(y_pred, y, loss_mask, mean, std)

        stats = {'MAE_metric': float(loss.item())}

        return stats

    def inference_one_epoch(self, epoch, phase):
        assert phase in ['train', 'val']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = len(self.loader[phase])

        for param in self.model.parameters():
            param.grad = None

        for c_iter, batch in enumerate(self.loader[phase]):
            # Forward pass
            stats = self.inference_one_batch(batch, phase)

            # Run optimization
            if phase == 'train':
                self.optimizer.step()

                # Clear gradients
                for param in self.model.parameters():
                    param.grad = None

            # update to stats_meter
            for key, value in stats.items():
                stats_meter[key].update(value)

            if phase == 'train' and (c_iter + 1) % self.freq_average_train_loss == 0:
                curr_iter = num_iter * epoch + (c_iter + 1)
                message = f'{phase}:\tEpoch: {epoch} [{c_iter + 1}/{num_iter}]\t'

                for key, value in stats_meter.items():
                    self.writer.add_scalar(f"train/{key}", value.avg, curr_iter)
                    message += f'{key}: {value.avg:.6f}\t'
                    stats_meter[key].reset()

                self.logger.info(message)
                self.writer.add_scalar("train/learning_rate", self._get_lr(), curr_iter)

        return stats_meter

    @staticmethod
    def stats_dict():
        stats = dict()
        stats['MAE_metric'] = 0.
        return stats

    def stats_meter(self):
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def train(self):
        self.logger.info('Start training...\n')
        start_time = time.time()

        for epoch in range(self.start_epoch, self.n_epochs):
            print_msg = f'Epoch {epoch}/{self.n_epochs - 1}'
            self.logger.info('\n{}\n{}\n'.format(print_msg, '-' * len(print_msg)))

            # ---------------------------------------------------------------------------- #
            # TRAINING
            # ---------------------------------------------------------------------------- #
            train_stats_meter = self.inference_one_epoch(epoch, 'train')

            # ---------------------------------------------------------------------------- #
            # VALIDATION
            # ---------------------------------------------------------------------------- #
            if (epoch + 1) % self.evaluate_rate == 0:
                val_stats_meter = self.inference_one_epoch(epoch, 'val')

                message = f"\nval:\tEpoch: {epoch}\t\t"

                for key, value in val_stats_meter.items():
                    self.writer.add_scalar(f"val/{key}", value.avg, epoch)
                    message += f'{key}: {value.avg:.6f}\t'
                self.logger.info(message + '\n')
                self.writer.add_scalar("val/learning_rate", self._get_lr(), epoch)

                # Save the best model
                if val_stats_meter['MAE_metric'].avg < self.best_loss:
                    self.best_loss = val_stats_meter['MAE_metric'].avg
                    self.index_best_loss = epoch

                    self._save_checkpoint(epoch, train_stats_meter['MAE_metric'].avg,
                                          val_stats_meter['MAE_metric'].avg, self.path_model_best)

                    # tensorboard logging: hyper-parameters
                    self.writer.add_hparams(hparam_dict=self.hparams,
                                            metric_dict={'hparam/MAE_metric': val_stats_meter['MAE_metric'].avg},
                                            run_name=self.tboard_log_dir)

                # After the epoch if finished, update the learning rate scheduler
                if self.scheduler is not None:
                    if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(val_stats_meter['MAE_metric'].avg)
                    else:
                        self.scheduler.step()

            # Save the model at the selected interval
            if (epoch + 1) % self.save_model_rate == 0 and epoch > self.evaluate_rate:
                name = 'Model_after_' + str(epoch + 1) + '_epochs.pth'
                self._save_checkpoint(epoch, train_stats_meter['MAE_metric'].avg, val_stats_meter['MAE_metric'].avg,
                                      os.path.join(self.checkpoint_dir, name))

        time_elapsed = time.time() - start_time
        time_text = time.strftime('%H:%M:%S', time.gmtime(time_elapsed))

        self.logger.info(f'\n\nTraining finished!\nTraining time: {time_text}')
        self.logger.info(f'\nBest model at epoch: {self.index_best_loss}')
        self.logger.info('Validation loss of the best model: {:.6f}'.format(self.best_loss))
        self.writer.close()

        # Save the last model
        self._save_checkpoint(epoch, train_stats_meter['MAE_metric'].avg, val_stats_meter['MAE_metric'].avg,
                              self.path_model_last)
