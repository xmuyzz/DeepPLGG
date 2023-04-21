import os
import numpy as np
import pandas as pd
from datetime import datetime
from time import localtime, strftime


def tr_logger(log_dir, cls_task, cnn_model, epoch, batch_size, lr, y_va):
    n_va = y_va.shape[0]
    n_tr = n_va*9
    log_path = log_dir + '/train_log_' + strftime('%Y_%m_%d', localtime()) + '.txt'
    with open(log_path, 'w') as f:
        #f.write('\n-------------------------------------------------------------------')
        f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
        #f.write('\n%s:' % strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        f.write('\nclassification task: %s' % cls_task)
        f.write('\nCNN model: %s' % cnn_model)
        f.write('\ntrain size: %s' % n_tr)
        f.write('\nval size: %s' % n_va)
        f.write('\ninitial lr: %s' % lr)
        f.write('\nepoch: %s' % epoch)
        f.write('\nbatch size: %s' % batch_size)
        f.write('\n')
        f.close()
    #print('successfully save train logs.')


def cb_logger(log_dir, tr_loss, va_loss, va_auc, epoch, lr): 
    #save_path = save_dir + '/train_log_' + strftime('%Y_%m_%d_%H_%M_%S', localtime()) + '.txt'
    log_path = log_dir + '/test_log_' + strftime('%Y_%m_%d', localtime()) + '.txt'
    with open(log_path, 'a') as f:
        #f.write('\n-------------------------------------------------------------------')
        #f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
        #f.write('\ntraining start......')
        f.write('\n%s:' % strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
        f.write('\nepoch: %s' % epoch)
        f.write('\ntr loss: %s' % tr_loss)
        f.write('\nva loss: %s' % va_loss)
        f.write('\nva AUC: %s' % va_auc)
        f.write('\nlr: %s' % lr)
        f.write('\n')
        f.close()
    #print('successfully save train logs.')


def test_logger(cls_task, run_type, save_dir, cms, cm_norms, reports, prc_aucs, 
              roc_stats, cnn_model, saved_model, epoch, batch_size, lr): 
    # va_dir = proj_dir + '/log/' + cls_task + '/val'
    # ts_dir = proj_dir + '/log/' + cls_task + '/test'
    # tx_dir = proj_dir + '/log/' + cls_task + '/external'
    # if not os.path.exists(va_dir):
    #     os.makedirs(va_dir)
    # if not os.path.exists(ts_dir):
    #     os.makedirs(ts_dir)
    # if not os.path.exists(tx_dir):
    #     os.makedirs(tx_dir)

    if run_type == 'val':
        log_fn = 'va_logs.text'
    elif run_type == 'test':
        log_fn = 'ts_logs.text'
    elif run_type == 'external':
        log_fn = 'tx_logs.text'
        
    save_path = os.path.join(save_dir, log_fn)
    if cls_task == 'tumor':
        with open(save_path, 'a') as f:
            f.write('\n------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\nprc image: %s' % prc_aucs[0])
            f.write('\nroc image:\n %s' % roc_stats[0])
            f.write('\ncm image:\n %s' % cms[0])
            f.write('\ncm image:\n %s' % cm_norms[0])
            f.write('\nreport image:\n %s' % reports[0])
            f.write('\ncnn model: %s' % cnn_model)
            f.write('\nsaved model: %s' % saved_model)
            f.write('\nepoch: %s' % epoch)
            f.write('\nlearning rate: %s' % lr)
            f.write('\nbatch size: %s' % batch_size)
            f.write('\n')
            f.close()    
    else:
        with open(save_path, 'a') as f:
            f.write('\n------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\nprc image: %s' % prc_aucs[0])
            f.write('\nprc patient prob: %s' % prc_aucs[1])
            f.write('\nprc patient pos: %s' % prc_aucs[2])
            f.write('\nroc image:\n %s' % roc_stats[0])
            f.write('\nroc patient prob:\n %s' % roc_stats[1])
            f.write('\nroc patient pos:\n %s' % roc_stats[2])
            f.write('\ncm image:\n %s' % cms[0])
            f.write('\ncm image:\n %s' % cm_norms[0])
            f.write('\ncm patient prob:\n %s' % cms[1])
            f.write('\ncm patient prob:\n %s' % cm_norms[1])
            f.write('\ncm patient pos:\n %s' % cms[2])
            f.write('\ncm patient pos:\n %s' % cm_norms[2])
            f.write('\nreport image:\n %s' % reports[0])
            f.write('\nreport patient prob:\n %s' % reports[1])
            f.write('\nreport patient pos:\n %s' % reports[2])
            f.write('\ncnn model: %s' % cnn_model)
            f.write('\nsaved model: %s' % saved_model)
            f.write('\nepoch: %s' % epoch)
            f.write('\nlearning rate: %s' % lr)
            f.write('\nbatch size: %s' % batch_size)
            f.write('\n')
            f.close()  

    print('successfully save logs.')
    
