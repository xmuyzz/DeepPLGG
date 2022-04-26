import numpy as np
import os
import glob
import pickle
import pandas as pd
import nibabel as nib
from sklearn.model_selection import KFold, train_test_split
from datasets.resize_3d import resize_3d
from opts import parse_opts



def pat_data(curation_dir):

    # labels
    df = pd.read_csv(os.path.join(curation_dir, 'BRAF_slice.csv'))
    df = df[~df['BRAF-Status'].isin(['In Review'])]
    labels = []
    img_dirs = []
    for braf in df['BRAF-Status']:
        if braf == 'No BRAF mutation':
            label = 0
            #print(braf)
        else:
            label = 1
            #print(braf)
        labels.append(label)
    df['label'] = labels

    # train test split
    df_train, df_test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=0, 
        stratify=df[['label']])
    df_train, df_val = train_test_split(
        df_train,
        test_size=0.1,
        random_state=1234,
        stratify=df_train[['label']])
    print(df_train.shape)
    print(df_val.shape)
    print(df_test.shape)
    print('train:', df_train['label'].value_counts())
    print('val:', df_val['label'].value_counts())
    print('test:', df_test['label'].value_counts())
    
    return df_train, df_val, df_test


def img_data(pro_data_dir, df, fn_arr_1ch, fn_arr_3ch, fn_df, channel, save_nii, nii_dir):

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
    for img_dir, pat_id, wmin, wmax in zip(df['img_dir'], df['Subject_ID'], df['wmin'], df['wmax']):
        count += 1
        print(count)
        img = resize_3d(
            img_dir=img_dir, 
            interp_type='nearest_neighbor',
            resize_shape=(192, 192))
        slice_range = range(wmin, wmax+1)
        data = img[slice_range, :, :]
        print('data shape:', data.shape)
        ### normalize signlas to [0, 1]
        data = np.interp(data, (data.min(), data.max()), (0, 1))
        if save_nii:
            nii = nib.Nifti1Image(data, affine=np.eye(4))
            fn = str(pat_id) + '.nii.gz'
            nib.save(nii, os.path.join(nii_dir, fn))
        ## stack all image arrays to one array for CNN input
        arr = np.concatenate([arr, data], 0)
        ### create patient ID and slice index for img
        slice_numbers.append(data.shape[0])
        for i in range(data.shape[0]):
            img = data[i, :, :]
            fn = pat_id + '_' + 'slice%s'%(f'{i:03d}')
            list_fn.append(fn)
    print('slice numbers:', slice_numbers)
    ### covert 1 channel input to 3 channel inputs for CNN
    if channel == 1:
        img_arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], 1)
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_1ch), img_arr)
    elif channel == 3:
        img_arr = np.broadcast_to(arr, (3, arr.shape[0], arr.shape[1], arr.shape[2]))
        img_arr = np.transpose(img_arr, (1, 2, 3, 0))
        #img_arr = np.transpose(img_arr, (1, 0, 2, 3))
        print('img_arr shape:', img_arr.shape)
        np.save(os.path.join(pro_data_dir, fn_arr_3ch), img_arr)
    
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
    img_df.to_csv(os.path.join(pro_data_dir, fn_df))
    print('data size:', img_df)
    print(img_df['label'].value_counts())


def get_img_dataset(pro_data_dir, df_train, df_val, df_test, channel, save_nii, nii_dir):

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
    
    dfs = [df_train, df_val, df_test]
    fns_arr_1ch = ['train_arr_1ch.npy', 'val_arr_1ch.npy', 'test_arr_1ch.npy']
    fns_arr_3ch = ['train_arr_3ch.npy', 'val_arr_3ch.npy', 'test_arr_3ch.npy']
    fns_df = ['train_img_df.csv', 'val_img_df.csv', 'test_img_df.csv']

    for df, fn_arr_1ch, fn_arr_3ch, fn_df in zip(dfs, fns_arr_1ch, fns_arr_3ch, fns_df):
        img_data(
            pro_data_dir=pro_data_dir,
            df=df,
            fn_arr_1ch=fn_arr_1ch,
            fn_arr_3ch=fn_arr_3ch,
            fn_df=fn_df,
            channel=channel,
            save_nii=save_nii,
            nii_dir=nii_dir)


if __name__ == '__main__':

    opt = parse_opts()
    if opt.root_dir is not None:
        opt.curation_dir = os.path.join(opt.root_dir, opt.curation)
        opt.pro_data_dir = os.path.join(opt.root_dir, opt.pro_data)
        opt.nii_dir = os.path.join(opt.root_dir, opt.nii)
        if not os.path.exists(opt.pro_data_dir):
            os.makedirs(opt.pro_data_dir)
        if not os.path.exists(opt.curation_dir):
            os.makedirs(opt.curation_dir)
        if not os.path.exists(opt.nii_dir):
            os.makedirs(opt.nii_dir)
    else:
        print('provide root dir to start!')

    df_train, df_val, df_test = pat_data(curation_dir=opt.curation_dir)

    get_img_dataset(
        pro_data_dir=opt.pro_data_dir, 
        df_train=df_train,
        df_val=df_val,
        df_test=df_test, 
        channel=opt.channel,
        save_nii=opt.save_nii,
        nii_dir=opt.nii_dir)


