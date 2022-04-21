import numpy as np
import os
import glob
import pickle
import pandas as pd
import nibabel as nib
from sklearn.model_selection import KFold, train_test_split
import resize_3d



def pat_data():

    # labels
    df = pd.read_csv(os.path.join(curation_dir, 'BRAF_slice.py'))
    df = df[~df['BRAF-Status'].isin(['data in review'])]
    labels = []
    img_dirs = []
    for braf in zip(df['BRAF-Status']):
        if braf == 'No BRAF mutation':
            label = 0
        else:
            label = 1
        labels.append(label)
    df['label'] = labels

    # train test split
    df_train, df_test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=0, 
        stratify=df[['label']])
#    data = df['img_dir']
#    label = df['label']
#    ID = df['Subject_ID']
#    img_train, img_test, label_train, label_test, ID_train, ID_test = train_test_split(
#        data,
#        label,
#        ID,
#        stratify=label,
#        test_size=0.3,
#        random_state=42)
    
    return df_train, df_test


def img_data(pro_data_dir, df, fn_arr_1ch, fn_arr_3ch, fn_df, channel=1):

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

    for img_dir, pat_id, wmin, wmax in zip(df['img_dir'], df['Subjec_ID'], df['wmin'], df['wmax']):
        count += 1
        img = resize_3d(
            img_dir=img_dir, 
            interp_type=interp_type,
            output_size=output_size
            )
        slice_range = [wmin, wmax]
        data = img[slice_range, :, :]
        ### normalize signlas to [0, 1]
        data = np.interp(data, (data.min(), data.max()), (0, 1))
        ## stack all image arrays to one array for CNN input
        arr = np.concatenate([arr, data], 0)
        ### create patient ID and slice index for img
        slice_numbers.append(data.shape[0])
        for i in range(data.shape[0]):
            img = data[i, :, :]
            fn = patient_id + '_' + 'slice%s'%(f'{i:03d}')
            list_fn.append(fn)

    ### covert 1 channel input to 3 channel inputs for CNN
    if channel == 1:
        img_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_1ch), img_arr)
    elif channel == 3:
        img_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img_arr = np.transpose(img_arr, (1, 2, 3, 0))
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_3ch), img_arr)
    
    # generate labels for CT slices
    list_label = []
    list_img = []
    for label, slice_number in zip(labels, slice_numbers):
        list_1 = [label] * slice_number
        list_label.extend(list_1)
    ### makeing dataframe containing img dir and labels
    img_df = pd.DataFrame({'fn': list_fn, 'label': list_label})
    pd.options.display.max_columns = 100
    pd.set_option('display.max_rows', 500)
    #print(img_df[0:100])
    img_df.to_csv(os.path.join(pro_data_dir, fn_df))
    #print('data size:', img_df.shape[0])


def get_img_dataset(pro_data_dir, df_train, df_test, channel):

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
    
    dfs = [df_train, df_test]
    fns_arr_1ch = ['train_arr_1ch.npy', 'test_arr_1ch.npy']
    fns_arr_3ch = ['train_arr_3ch.npy', 'test_arr_3ch.npy']
    fns_df = ['train_img_df.csv', 'test_img_df.csv']

    for df, fn_arr_1ch, fn_arr_3ch, fn_df in zip(dfs, fns_arr_1ch, fns_arr_3ch, fns_df):

        img_dataset(
            pro_data_dir=pro_data_dir,
            df=df,
            fn_arr_1ch=fn_arr_1ch,
            fn_arr_3ch=fn_arr_3ch,
            fn_df=fn_df,
            channel=channel
            )



