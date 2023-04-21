import pydicom as dicom
import pandas as pd
import os
import numpy as np
import tarfile, zipfile
import glob
from io import BytesIO
from zipfile import ZipFile
import requests

def BCH_dcm_header():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/BCH_raw'
    df0 = pd.DataFrame(columns=['name'])
    dfs = []
    IDs = []
    count = 0
    for f in os.listdir(proj_dir + '/BCH_TOT'):
        if not f.startswith('.'):
            print(f)
            for img_dir in glob.glob(proj_dir + '/BCH_TOT/' + f + '/*/*'):
                key = img_dir.split('/')[-1].split('_')
                if 'T1' in key and 'SAG' not in key and 'COR' not in key:
                    count += 1
                    print(count, key)
                    dcms = [i for i in glob.glob(img_dir + '/*dcm')]
                    ds = dicom.read_file(dcms[0], force=True) 
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
                    #df.reset_index(inplace=True)
                    df = df.loc[~df.index.duplicated(keep='first')].reset_index()
                    #print('df:', df)
                    IDs.append(f)
                    dfs.append(df)
        #print(df)
    pd.set_option("display.max_rows", 100, "display.max_columns", 100)
    #print('dfs:', dfs)
    df = pd.concat([df.set_index('name') for df in dfs], axis=1, join='inner').reset_index()
    df.columns = ['name'] + IDs
    df = df.set_index('name').T.reset_index()
    df.to_csv(proj_dir + '/csv_file/dcm_header.csv', index=False)


def unzip_dicom():
    csv_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/curation/dicom_header'
    df = pd.read_csv(csv_dir + '/curation_data.csv')
    #df = df[df['seq_id'] == 'T2W']
    df = df[df['seq_id'] == 'T1W']
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


def CBTN_dcm_header():
    csv_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/curation/dicom_header'
    df = pd.read_csv(csv_dir + '/curation_data.csv')
    #df = df[df['seq_id'] == 'T2W']
    df = df[df['seq_id'] == 'T1W']
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
    df.to_csv(csv_dir + '/CBTN_T1W_dcm.csv', index=False)

def MRI_protocol():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/clinical_data'
    df1 = pd.read_csv(proj_dir + '/BCH_T1W_dcm.csv')
    df2 = pd.read_csv(proj_dir + '/CBTN_T1W_dcm.csv')
    #df1['scan'] = df1['Manufacturer'] + df1["Manufacturer's Model Name"] + df1['Magnetic Field Strength']
    df1['scan'] = df1['Manufacturer'] + df1["Manufacturer's Model Name"]
    scans = set(df1['scan'].to_list())
    print(scans)



if __name__ == '__main__':

    #BCH_dcm_header()
    #CBTN_dcm_header()
    #unzip_dicom()
    #rename_dir(data_dir)
    #unzip_dicom(csv_dir) 
    MRI_protocol()

