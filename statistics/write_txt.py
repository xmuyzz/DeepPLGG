import os
import numpy as np
import pandas as pd
from datetime import datetime
from time import localtime, strftime

def write_txt(run_type, save_dir, loss, acc, cm1, cm2, cm3, cm_norm1, cm_norm2, cm_norm3, 
              report1, report2, report3, prc_auc1, prc_auc2, prc_auc3, stat1, stat2, stat3, 
              run_model, saved_model, epoch, batch_size, lr): 

    if run_type == 'train':
        log_fn = 'train_logs.text'
        write_path = os.path.join(save_dir, log_fn)
        with open(write_path, 'a') as f:
            f.write('\n-------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\nval acc: %s' % acc)
            f.write('\nval loss: %s' % loss)
            f.write('\nrun model: %s' % run_model)
            f.write('\nsaved model: %s' % saved_model)
            f.write('\nepoch: %s' % epoch)
            f.write('\nlearning rate: %s' % lr)
            f.write('\nbatch size: %s' % batch_size)
            f.write('\n')
            f.close()
        print('successfully save train logs.')
    else:
        if run_type == 'val':
            log_fn = 'val_logs.text'
        elif run_type == 'test':
            log_fn = 'test_logs.text'
        elif run_type == 'exval':
            log_fn = 'exval_logs.text'
        elif run_type == 'exval2':
            log_fn = 'exval2_logs.text'
        
        write_path = os.path.join(save_dir, log_fn)
        with open(write_path, 'a') as f:
            f.write('\n------------------------------------------------------------------')
            f.write('\ncreated time: %s' % strftime('%Y-%m-%d %H:%M:%S', localtime()))
            f.write('\nval accuracy: %s' % acc)
            f.write('\nval loss: %s' % loss)
            f.write('\nprc image: %s' % prc_auc1)
            f.write('\nprc patient prob: %s' % prc_auc2)
            f.write('\nprc patient pos: %s' % prc_auc3)
            f.write('\nroc image:\n %s' % stat1)
            f.write('\nroc patient prob:\n %s' % stat2)
            f.write('\nroc patient pos:\n %s' % stat3)
            f.write('\ncm image:\n %s' % cm1)
            f.write('\ncm image:\n %s' % cm_norm1)
            f.write('\ncm patient prob:\n %s' % cm2)
            f.write('\ncm patient prob:\n %s' % cm_norm2)
            f.write('\ncm patient pos:\n %s' % cm3)
            f.write('\ncm patient pos:\n %s' % cm_norm3)
            f.write('\nreport image:\n %s' % report1)
            f.write('\nreport patient prob:\n %s' % report2)
            f.write('\nreport patient pos:\n %s' % report3)
            f.write('\nrun model: %s' % run_model)
            f.write('\nsaved model: %s' % saved_model)
            f.write('\nepoch: %s' % epoch)
            f.write('\nlearning rate: %s' % lr)
            f.write('\nbatch size: %s' % batch_size)
            f.write('\n')
            f.close()    
        print('successfully save logs.')
    
