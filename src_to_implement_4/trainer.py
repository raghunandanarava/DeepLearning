import torch as t
from sklearn.metrics import f1_score
# from tqdm.autonotebook import tqdm
import numpy as np

import time


class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=False,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss

        self._optim.zero_grad()
        with t.set_grad_enabled(True):
            pred = self._model(x)
            loss = self._crit(pred, y)
            loss.backward()
            self._optim.step()
        return loss

    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        pred = self._model(x)
        loss = self._crit(pred, y)
        return loss, pred

    def train_epoch(self, scheduler):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        self._model.train()
        loss = []

        # batch mal image and label
        for image, label in self._train_dl:
            device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
            image = image.to(device)
            label = label.to(device)

            loss.append(self.train_step(image, label).item())
        scheduler.step()

        avg_loss = np.mean(loss)
        # avg_loss2 = np.mean(loss2)
        print('Training set: The total L2 oss is {:.5f}, '.format(avg_loss))
        return avg_loss

    def val_test(self):
        # set eval mode
        # disable gradient computation
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        self._model.eval()
        val_loss = []

        #to compute final f1
        labels_crack = []
        labels_inac = []
        preds_crack = []
        preds_inac = []
        f1s_1 = []
        f1s_2 = []
        with t.no_grad():
            for image, label in self._val_test_dl:
                device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
                image = image.to(device)
                label = label.to(device)

                loss1, pred1 = self.val_test_step(image, label)

                pred1 = pred1.detach().cpu().numpy()
                pred_class1 = np.where(pred1[:,0] > 0.5, 1.0, 0.0)
                pred_class2 = np.where(pred1[:, 1] > 0.5, 1.0,0.0)
                preds_crack += list(pred_class1)
                preds_inac += list(pred_class2)
                #print('the pred for crack is ')
                #print(pred_class1)
                # pred_class1 = pred_class1.detach().cpu().numpy().flatten()
                # pred_class2 = pred_class2.detach().cpu().numpy().flatten()

                label = label.detach().cpu().numpy()
                label1 = list(label[:, 0])
                label2 = list(label[:, 1])
                labels_crack += label1
                labels_inac += label2
                #print('the label is')
                #print(label1)

                # f1_1 = f1_score(label1, pred_class1)
                # f1_2 = f1_score(label2, pred_class2)
                val_loss.append(loss1.item())
                # val_loss2.append(loss2.item())
                # f1s_1.append(f1_1)
                # f1s_2.append(f1_2)

        avg_loss = np.mean(val_loss)
        f1_crack = f1_score(labels_crack, preds_crack)
        f1_inactive = f1_score(labels_inac, preds_inac)

        print(
            'Validation set: The L2 Loss is {:.5f}, The f1 score of the crack prediction is {:.5f} '
            'and The f1 score of the inactive prediction is {:.5f}'.format(
                avg_loss, f1_crack, f1_inactive))
        return avg_loss, f1_crack, f1_inactive

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch

        epoch = 0
        waiting_count = 0
        train_loss1 = []

        val_loss1 = []

        f1_metric_crack = []
        f1_metric_inactive = []
        lr_scheduler = t.optim.lr_scheduler.StepLR(optimizer=self._optim, step_size=15, gamma=0.5)
        #lr_scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer=self._optim, gamma=0.1)


        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation

            min_valloss = 0
            if len(val_loss1) > 0:
                min_valloss = np.min(val_loss1)
            since = time.time()
            total_time = 0
            epoch += 1
            print('Epoch {}'.format(epoch))
            for param_group in self._optim.param_groups:
                lr = param_group['lr']
                print('the learning rate now is {}'.format(lr))

            training_loss1 = self.train_epoch(lr_scheduler)
            train_loss1.append(training_loss1)
            # train_loss2.append(training_loss2)
            val_loss, f1_crack, f1_inactive = self.val_test()
            val_loss1.append(val_loss)

            f1_metric_crack.append(f1_crack)
            f1_metric_inactive.append(f1_inactive)
            self.save_checkpoint(epoch)

            # lr_scheduler.step()
            time_elapsed = time.time() - since
            total_time += time_elapsed
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('-' * 10)



            if val_loss < min_valloss:
                waiting_count = 0
            else:
                waiting_count += 1

            if waiting_count >= self._early_stopping_patience:
                break
        sum = np.asarray(f1_metric_crack) + np.asarray(f1_metric_inactive)
        num_epoch = np.argmax(sum)
        max = np.max(sum) / 2
        print(
            "The average f1 score of crack prediction is {:.5f}, of inactive prediction is {:.5f}, the max value of all is {:.5f} "
            "and the corresponding epoch is {}".format(
                np.mean(f1_metric_crack),
                np.mean(f1_metric_inactive),
                max, num_epoch + 1))

        return train_loss1, val_loss1
