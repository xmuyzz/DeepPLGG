import os
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import keras
import numpy as np
from matplotlib.pylab import plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_auc_score
from logger import cb_logger


class my_callback(Callback):
    def __init__(self, model, log_dir, x_va, y_va):
        self.cnn_model = model
        self.log_dir = log_dir
        self.x_va = x_va
        self.y_va = y_va
        self.tr_losses = []
        self.va_losses = []
        self.va_aucs = []
        self.lrs = []
        self.best_va_loss = 1
        self.best_va_auc = 0.9
        
    def on_train_batch_end(self, batch, logs=None):
        tr_loss = logs['loss']
        #self.tr_losses.append(tr_loss)

    def on_epoch_end(self, epoch, logs=None):
        tr_loss = np.round(logs['loss'], 3)
        va_loss = np.round(logs['val_loss'], 3)
        self.tr_losses.append(tr_loss)
        self.va_losses.append(va_loss)
        lr = float(K.get_value(self.model.optimizer.lr))
        lr = round(lr, 6)
        # lr = self.model.optimizer.lr
        self.lrs.append(lr)
        #label = label.reshape(1, *label.shape)
        #pred = self.cnn_model.predict(image.reshape(1, *image.shape))
        #pred = np.squeeze(pred)
        pred = self.cnn_model.predict(self.x_va)
        pred = np.squeeze(pred)
        #print('pred:', pred.shape)
        #print('label:', self.y_va.shape)
        va_auc = round(roc_auc_score(self.y_va, pred), 3)
        self.va_aucs.append(va_auc)
        #print('epcoh:', epoch)
        #print('tr loss:', round(tr_loss, 3), 'va loss:', round(va_loss, 3))
        #print('val auc:', va_auc)
        
        # save best AUC and loss model
        if va_loss < self.best_va_loss:
            self.cnn_model.save(self.log_dir + '/best_loss_model.h5')
            self.best_va_loss = va_loss
            print('best loss model saved.')
        elif va_auc > self.best_va_auc:
            self.cnn_model.save(self.log_dir + '/best_auc_model.h5')
            self.best_va_auc = va_auc
            print('best auc model saved.')
        
        # save logs for losses and aucs
        metric_dir = self.log_dir + '/metrics'
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
        np.save(metric_dir + '/tr_losses.npy', self.tr_losses)
        np.save(metric_dir + '/va_losses.npy', self.va_losses)
        np.save(metric_dir + '/lrs.npy', self.lrs)
        np.save(metric_dir + '/auc_scores.npy', self.va_aucs)

        # make log and update every epoch to check training progress
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #ax.set_aspect('equal')
        epoch = len(self.tr_losses) - 1
        plt.plot(self.tr_losses, color='red', linewidth=1, label='tr_loss')
        plt.plot(self.va_losses, color='green', linewidth=1, label='va_loss')
        plt.plot(self.va_aucs, color='blue', linewidth=1, label='va_auc')
        plt.xlim([0, epoch+1])
        plt.ylim([0, 1])
        #ax.axhline(y=0, color='k', linewidth=4)
        #ax.axhline(y=1.03, color='k', linewidth=4)
        #ax.axvline(x=-0.03, color='k', linewidth=4)
        #ax.axvline(x=1, color='k', linewidth=4)
        if epoch < 20:
            interval = 1
        elif epoch >= 20 and epoch < 50:
            interval = 5
        elif epoch >= 50:
            interval = 10
        x = np.arange(0, epoch+1, interval, dtype=int).tolist()
        plt.xticks(x, fontsize=8, fontweight='bold')
        plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=8, fontweight='bold')
        #plt.yticks(fontsize=8, fontweight='bold')
        plt.yticks(fontsize=8, fontweight='bold')
        plt.xlabel('EPOCH', fontweight='bold', fontsize=8)
        plt.ylabel('AUC/LOSS', fontweight='bold', fontsize=8)
        plt.legend(loc='upper right', prop={'size': 8, 'weight': 'bold'})
        plt.grid(True)
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        plt.savefig(self.log_dir + '/progress.png', format='png', dpi=200)
        plt.close()
        
        # save training loggers
        save_logger = True
        if save_logger:
            cb_logger(log_dir=self.log_dir, tr_loss=tr_loss, va_loss=va_loss, va_auc=va_auc, epoch=epoch, lr=lr)

    def on_train_end(self):
        self.cnn_model.save(self.model_dir + '/final_model.h5')   
        best_auc = max(self.va_aucs)
        print('best val AUC:', best_auc)

    

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)