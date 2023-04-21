
import os
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
from collections import Counter
from datetime import datetime
from time import localtime, strftime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from crop_image import crop_image
from resize_3d import resize_3d
import SimpleITK as sitk

#----------------------------------------------------------------
# error analysis on image level
#---------------------------------------------------------------
def error_img(run_type, val_save_dir, test_save_dir, input_channel, crop):
   
    ### load train data based on input channels
    if run_type == 'val': 
        file_dir = val_save_dir
        input_fn = 'df_val_pred.csv'
        output_fn = 'val_error_img.csv'
        col = 'y_val'
    elif run_type == 'test':
        file_dir = test_save_dir
        input_fn = 'df_test_pred.csv'
        output_fn = 'test_error_img.csv'
        col = 'y_test'
    
    ### dataframe with label and predictions
    df = pd.read_csv(os.path.join(file_dir, input_fn))
    print(df[0:10])
    df.drop([col, 'ID'], axis=1, inplace=True)
    df = df.drop(df[df['y_pred_class'] == df['label']].index)
    df[['y_pred']] = df[['y_pred']].round(3)
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    print(df[0:200]) 
    df.to_csv(os.path.join(file_dir, output_fn))
    df_error_img = df

    return df_error_img
#----------------------------------------------------
# generate images for error check
#---------------------------------------------------- 
def save_error_img(df_error_img, run_type, n_img, val_img_dir, test_img_dir, 
                   val_error_dir, test_error_dir):

    ### store all indices that has wrong predictions
    indices = []
    df = df_error_img
    for i in range(df.shape[0]):
        indices.append(i)
    print(x_val.shape)
    arr = x_val.take(indices=indices, axis=0)
    print(arr.shape)
    arr = arr[:, :, :, 0]
    print(arr.shape)
    arr = arr.reshape((arr.shape[0], 192, 192))
    print(arr.shape)
    
    ## load img data
    if run_type == 'val':
        if input_channel == 1:
            fn = 'val_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'val_arr_3ch.npy'
        x_val = np.load(os.path.join(data_pro_dir, fn))
        file_dir = val_error_dir
   
    elif run_type == 'test':
        if input_channel == 1:
            fn = 'test_arr_1ch.npy'
        elif input_channel == 3:
            fn = 'test_arr_3ch.npy'
        x_val = np.load(os.path.join(data_pro_dir, fn))
        file_dir = test_error_dir

    ### display images for error checks
    count = 0
    for i in range(n_img):
    # for i in range(arr.shape[0]):
        #count += 1
        #print(count)
        img = arr[i, :, :]
        fn = str(i) + '.jpeg'
        img_fn = os.path.join(file_dir, fn)
        mpl.image.imsave(img_fn, img, cmap='gray')
     
    print('save images complete!!!')

#----------------------------------------------------------------------
# error analysis on patient level
#----------------------------------------------------------------------
def error_patient(run_type, val_save_dir, test_save_dir, threshold):

    if run_type == 'val':
        file_dir = val_save_dir
        input_fn = 'df_val_pred.csv'
        output_fn = 'val_error_patient.csv'
    elif run_type == 'test':
        file_dir = test_save_dir
        input_fn = 'df_test_pred.csv'
        output_fn = 'test_error_patient.csv'

    df_sum = pd.read_csv(os.path.join(file_dir, input_fn))
    df_mean = df_sum.groupby(['ID']).mean()
    y_true = df_mean['label']
    y_pred = df_mean['y_pred']
    y_pred_classes = []
    for pred in y_pred:
        if pred < threshold:
            y_pred_class = 0
        elif pred >= threshold:
            y_pred_class = 1
        y_pred_classes.append(y_pred_class)
    df_mean['y_pred_class_thr'] = y_pred_classes
    df = df_mean
    df[['y_pred', 'y_pred_class']] = df[['y_pred', 'y_pred_class']].round(3)
    df = df.drop(df[df['y_pred_class_thr'] == df['label']].index)
    #df.drop(['y_test'], inplace=True, axis=1)
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    print(df)
    df_error_patient = df
    df.to_csv(os.path.join(file_dir, output_fn))

#----------------------------------------------------------------------
# save error scan
#----------------------------------------------------------------------    
def save_error_patient(run_type, val_save_dir, test_save_dir, val_error_dir, test_error_dir, norm_type,
                       crop_shape, return_type1, return_type2, interp_type, output_size):
    count = 0
    dirs_error = []
    patient_ids = []
    
    if run_type == 'val':
        x_val = pd.read_pickle(os.path.join(val_img_dir, 'x_val.p'))
        save_dir = val_error_dir
        df = pd.read_csv(os.path.join(val_save_dir, 'val_error_patient.csv'))
        ## find file dirs for error patients
        for val_file in x_val:
            count += 1
            ### create consistent patient ID format
            if val_file.split('/')[-1].split('_')[0] == 'PMH':
                patient_id = 'PMH' + val_file.split('/')[-1].split('-')[1][2:5].strip()
            elif val_file.split('/')[-1].split('-')[1] == 'CHUM':
                patient_id = 'CHUM' + val_file.split('/')[-1].split('_')[1].split('-')[2].strip()
            elif val_file.split('/')[-1].split('-')[1] == 'CHUS':
                patient_id = 'CHUS' + val_file.split('/')[-1].split('_')[1].split('-')[2].strip()
            if patient_id in df['ID'].to_list():
                patient_ids.append(patient_id)
                print('error scan:', patient_id)
                dirs_error.append(val_file)
   
    elif run_type == 'test':
        save_dir = test_error_dir
        df = pd.read_csv(os.path.join(test_save_dir, 'test_error_patient.csv')) 
        for test_file in sorted(glob.glob(mdacc_data_dir + '/*nrrd')):
            patient_id = 'MDACC' + test_file.split('/')[-1].split('-')[2][1:4].strip()
            if patient_id in df['ID'].to_list():
                patient_ids.append(patient_id)
                print('error scan:', patient_id)
                dirs_error.append(test_file)
    
    for file_dir, patient_id in zip(dirs_error, patient_ids):
        if crop == True:
            ## crop image from (512, 512, ~160) to (192, 192, 110)
            img_crop = crop_image(
                nrrd_file=file_dir,
                crop_shape=crop_shape,
                return_type=return_type1,
                output_dir=None
                )
            img_nrrd = img_crop
        elif crop == False:
            img_nrrd = sitk.ReadImage(file_dir)
        
        ### sitk axis order (x, y, z), np axis order (z, y, x)
        resized_img = resize_3d(
            img_nrrd=img_nrrd,
            interp_type=interp_type,
            output_size=output_size,
            patient_id=patient_id,
            return_type=return_type2,
            save_dir=save_dir
            )
        data = resized_img[0:32, :, :]
        data[data <= -1024] = -1024
        data[data > 700] = 0
        if norm_type == 'np_interp':
            arr_img = np.interp(data, [-200, 200], [0, 1])
        elif norm_type == 'np_clip':
            arr_img = np.clip(data, a_min=-200, a_max=200)
            MAX, MIN = arr_img.max(), arr_img.min()
            arr_img = (arr_img - MIN) / (MAX - MIN)
        ## save npy array to image 
        img = sitk.GetImageFromArray(arr_img)
        fn = str(patient_id) + '.nrrd'
        sitk.WriteImage(img, os.path.join(save_dir, fn))
    print('save error scans!')

    ## save error scan as nrrd file
        
#----------------------------------------------------------------------------
# main funtion
#---------------------------------------------------------------------------
if __name__ == '__main__':

    val_img_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/val_img_dir'
    val_save_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/val'
    test_save_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test'
    val_error_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/val/error'
    test_error_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test/error'
    mdacc_data_dir  = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_mdacc'
    input_channel = 3
    crop = True
    n_img = 20
    save_img = False
    thr_prob = 0.5
    run_type = 'val'
    norm_type = 'np_interp'
    crop_shape = [192, 192, 110]
    return_type1 = 'nrrd'
    return_type2 = 'npy'
    interp_type = 'linear'
    output_size = (96, 96, 36)

    os.mkdir(val_error_dir)  if not os.path.isdir(val_error_dir)  else None
    os.mkdir(mdacc_data_dir) if not os.path.isdir(mdacc_data_dir) else None
    os.mkdir(val_img_dir)    if not os.path.isdir(val_img_dir)    else None
    os.mkdir(test_error_dir) if not os.path.isdir(test_error_dir) else None
    
    df = error_img(
        run_type=run_type,
        val_save_dir=val_save_dir,
        test_save_dir=test_save_dir,
        input_channel=input_channel,
        crop=crop
        )

    error_patient(
        run_type=run_type,
        val_save_dir=val_save_dir,
        test_save_dir=test_save_dir,
        threshold=thr_prob
        )

    save_error_patient(
        run_type=run_type, 
        val_save_dir=val_save_dir,
        test_save_dir=test_save_dir,
        val_error_dir=val_error_dir, 
        test_error_dir=test_error_dir, 
        norm_type=norm_type,
        crop_shape=crop_shape, 
        return_type1=return_type1, 
        return_type2=return_type2, 
        interp_type=interp_type, 
        output_size=output_size
        )
    
    if save_img == True:
        save_error_img(
            df=df,
            n_img=n_img,
            run_type=run_type,
            val_img_dir=val_img_dir,
            test_img_dir=test_img_dir,
            val_error_dir=vel_error_dir,
            test_error_dir=test_error_dir,
            )
