import pydicom as dicom
import pandas as pd
import os
import numpy as np
import tarfile, zipfile
import glob
from io import BytesIO
from zipfile import ZipFile
import requests


def rename_dir(data_dir):

    count = 0
    for root, subdirs, files in os.walk(data_dir):
        count += 1
        print(count)
        for fn in files:
            print(fn)
            old_path = os.path.join(root, fn)
            fn = fn.replace(' - ', '_')
            print(fn)
            new_path = os.path.join(root, fn)
            os.rename(old_path, new_path)


def unzip_dicom(csv_dir):

    df = pd.read_csv(csv_dir + '/curation_data.csv')
    df = df[df['seq_id'] == 'T2W']
    df.drop_duplicates(subset='pat_id', keep='first', inplace=True, ignore_index=True)
    print(df)
    fns = []
    dirs = []
    IDs = []
    for i, path in enumerate(df['path']):
        data_dir = os.path.split(path)[0]
        #print(data_dir)
        for dicom_dir in glob.glob(data_dir + '/*.dicom.zip'):
            if os.path.exists(dicom_dir):
                print(i, dicom_dir.split('/')[-7])
                output_dir = data_dir + '/dicom'
                if not output_dir:
                   os.makedirs(output_dir)
                zf = zipfile.ZipFile(dicom_dir, 'r')
                zf.extractall(output_dir)
                zf.close()
            else:
                ID = dicom_dir.split('/')[-7]
                IDs.append(ID)
                dirs.append(dicom_dir)
    print('missing data:', IDs, dirs)


def get_dicom_header(csv_dir):

    df = pd.read_csv(csv_dir + '/curation_data.csv')
    df = df[df['seq_id'] == 'T2W']
    df.drop_duplicates(subset='pat_id', keep='first', inplace=True, ignore_index=True)
    #print(df)
    df0 = pd.DataFrame(columns=['name'])
    dfs = []
    IDs = []
    for count, (path, pat_id) in enumerate(zip(df['path'], df['pat_id'])):
        print(count)
        data_dir = os.path.split(path)[0]
        dcms = [i for i in os.listdir(data_dir + '/dicom')]
        ds = dicom.read_file(data_dir + '/dicom/' + dcms[0], force=True) 
        #for i in ds:
        #    print(i)
        print(ds)
        #print(ds.values())
        df = pd.DataFrame(ds.values())
        df[0] = df[0].apply(lambda x: dicom.dataelem.DataElement_from_raw(x) 
                if isinstance(x, dicom.dataelem.RawDataElement) else x)
        df['name'] = df[0].apply(lambda x: x.name)
        df['value'] = df[0].apply(lambda x: x.value)
        df = df[['name', 'value']].set_index('name')
        df.drop('Pixel Data', axis=0, inplace=True)
        df.reset_index(inplace=True)
        IDs.append(pat_id)
        dfs.append(df)
        #print(df)
    #pd.set_option("display.max_rows", 100, "display.max_columns", 100)
    df = pd.concat([df.set_index('name') for df in dfs], axis=1, join='inner').reset_index()
    df.columns = ['name'] + IDs
    df = df.set_index('name').T.reset_index()
    df.to_csv(csv_dir + '/dicom_meta2.csv', index=False)


if __name__ == '__main__':

    csv_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/curation/dicom_header'
    proj_dir = '/mnt/aertslab/DATA/Glioma/flywheel_20210210_223349/flywheel/LGG/SUBJECTS'
    data_dir = proj_dir + '/C15867/SESSIONS/4687d_B_brain/ACQUISITIONS/3 - t2_tse3dvfl_tra_st2_p2_WIP/FILES'

    get_dicom_header(csv_dir)
    #rename_dir(data_dir)
    #unzip_dicom(csv_dir) 




