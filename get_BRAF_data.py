import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
from sklearn.model_selection import KFold, train_test_split
from datasets.resize_3d import resize_3d


def img_data(save_dir, df, fn_arr_1ch, fn_arr_3ch, fn_df, channel, save_nii, nii_dir):
    """
    get stacked image slices from scan level CT and corresponding labels and IDs;
    Args:
        run_type {str} -- train, val, test, external val, pred;
        pro_data_dir {path} -- path to processed data;
        nrrds {list} --  list of paths for CT scan files in nrrd format;
        IDs {list} -- list of patient ID;
        labels {list} -- list of patient labels;
        slice_range {np.array} -- image slice range in z direction for cropping;
        run_type {str} -- train, val, test, or external val;
        input_channel {str} -- image channel, default: 3;
    Returns:
        img_df {pd.df} -- dataframe contains preprocessed image paths, label, ID (image level);
    """
    # get image slice and save them as numpy array
    count = 0
    slice_numbers = []
    list_fn = []
    arr = np.empty([0, 192, 192])
    print(df)
    for img_path, pat_id, zmin, zmax in zip(df['img_path'], df['pat_id'], df['zmin'], df['zmax']):
        count += 1
        print(count, pat_id)
        img = resize_3d(
            img_dir=img_path, 
            interp_type='nearest_neighbor',
            resize_shape=(192, 192))
        # trim 2 slices on top and bottom to get rid of small lesions
        if zmax - zmin <= 4:
            slice_range = range(zmin, zmax+1)
        else:
            slice_range = range(zmin+2, zmax-1)
        data = img[slice_range, :, :]
        print('data shape:', data.shape)
        ### normalize signlas to [0, 1]
        data = np.interp(data, (data.min(), data.max()), (0, 1))
        if save_nii:
            nii = nib.Nifti1Image(data, affine=np.eye(4))
            nib.save(nii, nii_dir + '/' + str(pat_id) + '.nii.gz')
        ## stack all image arrays to one array for CNN input
        arr = np.concatenate([arr, data], 0)
        ### create patient ID and slice index for img
        slice_numbers.append(data.shape[0])
        for i in range(data.shape[0]):
            img = data[i, :, :]
            fn = str(pat_id) + '_' + 'slice%s'%(f'{i:03d}')
            list_fn.append(fn)
    print('slice numbers:', slice_numbers)
    ### covert 1 channel input to 3 channel inputs for CNN
    if channel == 1:
        img_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        print('img_arr shape:', img_arr.shape)
        np.save(save_dir + '/' + fn_arr_1ch, img_arr)
    elif channel == 3:
        img_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img_arr = np.transpose(img_arr, (1, 2, 3, 0))
        #img_arr = np.transpose(img_arr, (1, 0, 2, 3))
        print('img_arr shape:', img_arr.shape)
        np.save(save_dir + '/' + fn_arr_3ch, img_arr)
    
    # generate labels for CT slices
    list_label = []
    list_img = []
    for label, slice_number in zip(df['label'], slice_numbers):
        list_1 = [label] * slice_number
        list_label.extend(list_1)
    ### makeing dataframe containing img dir and labels
    img_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    #print(img_df[0:100])
    img_df.to_csv(save_dir + '/' + fn_df)
    print('data size:', img_df.shape[0])
    print(img_df['label'].value_counts())


def get_img_dataset(proj_dir, channel, save_nii):
    """
    Get np arrays for stacked images slices, labels and IDs for train, val, test dataset;
    Args:
        run_type {str} -- train, val, test, external val, pred;
        pro_data_dir {path} -- path to processed data;
        data_tot {list} -- list of data paths: ['data_train', 'data_val', 'data_test'];
        ID_tot {list} -- list of image IDs: ['ID_train', 'ID_val', 'ID_test'];
        label_tot {list} -- list of image labels: ['label_train', 'label_val', 'label_test'];
        slice_range {np.array} -- image slice range in z direction for cropping;
    """
    csv_dir = proj_dir + '/csv_file'
    save_dir = proj_dir + '/braf_cls'
    nii_dir = proj_dir + '/nii_data'

    # get data sets
    df_tx = pd.read_csv(csv_dir + '/CBTN.csv')
    df = pd.read_csv(csv_dir + '/BCH.csv')
    df_tr, df_ts = train_test_split(
        df, 
        test_size=0.2, 
        random_state=1234,
        stratify=df['label'])
    df_tr, df_va = train_test_split(
        df_tr,
        test_size=0.1,
        random_state=1234,
        stratify=df_tr['label'])
    print(df_tr.shape)
    print(df_va.shape)
    print(df_ts.shape)
    print(df_tx.shape)
    print('train:', df_tr['label'].value_counts())
    print('val:', df_va['label'].value_counts())
    print('external:', df_tx['label'].value_counts())

    dfs = [df_tr, df_va, df_ts, df_tx]
    fns_arr_1ch = ['tr_arr_1ch.npy', 'va_arr_1ch.npy', 'ts_arr_1ch.npy', 'tx_arr_1ch.npy']
    fns_arr_3ch = ['tr_arr_3ch.npy', 'va_arr_3ch.npy', 'ts_arr_3ch.npy', 'tx_arr_3ch.npy']
    fns_df = ['tr_img_df.csv', 'va_img_df.csv', 'ts_img_df.csv', 'tx_img_df.csv']
    for df, fn_arr_1ch, fn_arr_3ch, fn_df in zip(dfs, fns_arr_1ch, fns_arr_3ch, fns_df):
        img_data(
            save_dir=save_dir,
            df=df,
            fn_arr_1ch=fn_arr_1ch,
            fn_arr_3ch=fn_arr_3ch,
            fn_df=fn_df,
            channel=channel,
            save_nii=save_nii,
            nii_dir=nii_dir)


def get_BRAF_label(proj_dir):

    data_dir = proj_dir + '/braf_cls'
    fns_df = ['tr_img_df.csv', 'va_img_df.csv', 'ts_img_df.csv', 'tx_img_df.csv']
    for fn_df in fns_df:
        df = pd.read_csv(data_dir + '/' + fn_df)
        for cls_task in ['V600E', 'fusion', 'wild_type']:
            print('cls task:', cls_task)
            if cls_task == 'V600E':
                ys = []
                for label in df['label']:
                    if label == 2:
                        y = 1
                    else:
                        y = 0
                    ys.append(y)
                df['V600E'] = ys
            elif cls_task == 'fusion':
                ys = []
                for label in df['label']:
                    if label == 1:
                        y = 1
                    else:
                        y = 0
                    ys.append(y)
                df['fusion'] = ys
            elif cls_task == 'wild_type':
                ys = []
                for label in df['label']:
                    if label == 0:
                        y = 1
                    else:
                        y = 0
                    ys.append(y)
                df['wild_type'] = ys
        df.to_csv(data_dir + '/' + fn_df, index=False)
            
        
if __name__ == '__main__':

    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/BRAF'
    channel = 1

    #get_img_dataset(proj_dir=proj_dir, channel=channel, save_nii=False)

    get_BRAF_label(proj_dir)


