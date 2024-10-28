import gc
import numpy as np
import torch as t
from sklearn.metrics import f1_score, accuracy_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
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

        self.loss_list = []
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        if self._optim is not None:
            self._optim.zero_grad()

        y_output = self._model(x)

        calculated_loss = self._crit(y_output, y.to(t.float32))

        calculated_loss.backward()

        self._optim.step()

        return calculated_loss.item()

        
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions

        output_y = self._model(x)

        calc_loss = self._crit(output_y, y.to(t.float32))

        return calc_loss, output_y
        
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        
        self._model.train()
        epoch_loss = []
        batches = len(self._train_dl)

        loss = 0.0
        
        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            loss = self.train_step(x, y)

            epoch_loss.append(loss)

        avg_epoch_loss = sum(epoch_loss) / batches

        return avg_epoch_loss

    
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        
        self._model.eval()

        with t.no_grad():
            total_epoch_loss = []
            y_labels_true = []
            y_labels_pred = []

            batches = len(self._val_test_dl)

            for x, y_labels in enumerate(self._val_test_dl):
                if self._cuda:
                    x = x.cuda()
                    y_labels = y_labels.cuda()

                iter_loss, predicted_label = self.val_test_step(x, y_labels)
                
                y_labels_true.append(y_labels.cpu().numpy())    
                y_labels_pred.append(predicted_label.cpu().numpy())

                total_epoch_loss.append(iter_loss)                

            avg_epoch_loss = sum(total_epoch_loss) / batches

            y_labels_true = np.concatenate(y_labels_true)
            y_labels_pred = np.concatenate(y_labels_pred)

            calc_f1_score = f1_score(y_labels_true, (y_labels_pred>0.5).long(), average = "micro") 
            accuracy = accuracy_score(y_labels_true, (y_labels_pred>0.5).long())

            gc.collect()
            if self._cuda:
                t.cuda.empty_cache()

            print("F1 Score: ", calc_f1_score)
            print("Accuracy: ", accuracy)

            return avg_epoch_loss, calc_f1_score
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 

        train_losses = []
        val_losses = []
        epoch_counter = 0
        
        best_val_loss = np.inf
        best_f1_score = -1
        prev_loss = np.inf        

        pt = 0

        best_epoch = 0

        while True:
      
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation

            if epoch_counter >= epochs and epochs > 0:
                break
            
            print("Epoch: ", epoch_counter)
            train_loss = self.train_epoch()

            val_loss, f1_score = self.val_test()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                pt = 0

            if f1_score > best_f1_score:
                best_f1_score = f1_score
                self.save_checkpoint(epoch_counter)
                print("Saving Checkpoint.. ", epoch_counter)
                best_epoch = epoch_counter
            
            if val_loss > prev_loss:
                pt += 1
                if pt >= self._early_stopping_patience:
                    print("Early Stopping patience reached.. Saving the model")
                    break

            prev_loss = val_loss
            epoch_counter += 1

        self.restore_checkpoint(best_epoch)
        self.save_checkpoint(4001)
        self.save_onnx('trained_model.onnx')


        return train_losses, val_losses

        
                    
        
        
        
