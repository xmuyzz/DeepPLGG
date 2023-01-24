import os
import numpy as np
import glob
import pandas as pd
import pydicom as dicom


def get_clinical():

    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/clinical_data'
    data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/CBTN_BCH_Data/BCH'
    df0 = pd.read_csv(proj_dir + '/PLGG_DATA.csv')
    print(df0)
    print(df0.shape[0])

    MRNs = []
    last_names = []
    first_names = []
    MRIs = []
    count = 0
    MRI = 'yes'
    for dataset in ['BCH_curated', 'BCH_curated2', 'BCH_curated3']:
        for folder in os.listdir(data_dir + '/' + dataset):
            if not folder.startswith('.'):
                count += 1
                MRN = folder.split('_')[-1]
                last_name = folder.split('_')[0]
                first_name = folder.split('_')[1]
                print(count, MRN)
                MRNs.append(MRN)
                last_names.append(last_name)
                first_names.append(first_name)
                MRIs.append(MRI)
    df1 = pd.DataFrame({'BCH MRN': MRNs, 'LAST NAME': last_names, 'FIRST NAME': first_names, 'MRI': MRIs})
    df1.drop_duplicates(subset=['BCH MRN'], keep='first', inplace=True, ignore_index=True)
    print(df1)
    print('patient number:', df1.shape[0])
    df1.to_csv(proj_dir + '/BCH_MRI.csv')
    df1 = df1[['BCH MRN', 'MRI']]

    df0['BCH MRN'] = df0['BCH MRN'].astype(int)
    df1['BCH MRN'] = df1['BCH MRN'].astype(int)
    df = df0.merge(df1, on='BCH MRN', how='left').reset_index()
    df.drop_duplicates(subset=['BCH MRN'], keep='first', inplace=True, ignore_index=True)
    df = df[['BCH MRN', 'DFCI MRN', 'LAST NAME', 'FIRST NAME', 'DOB', 'MRI', 'Date of Path Diagnosis']]
    print(df)
    df.to_csv(proj_dir + '/BCH_MRI_clinical.csv')


def get_MRI_date():
    """ get MRI date from dicom header
    """
    data_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/CBTN_BCH_Data/BCH'
    MRNs = []
    dates = []
    bad_data = []
    for dataset in ['BCH_curated', 'BCH_curated2', 'BCH_curated3']:
        for root, dirs, files in os.walk(data_dir + '/' + dataset):
            if not dirs:
                if 'T2' in root:
                    #print(root)
                    dcm_dirs = [i for i in glob.glob(root + '/*dcm')]
                    ds = dicom.read_file(dcm_dirs[0])
                    ID = dcm_dirs[0].split('/')[-4]
                    print(ID)
                    try:
                        study_date = ds.StudyDate
                        dates.append(study_date)
                        MRNs.append(ID.split('_')[-1])
                    except Exception as e:
                        print(ID, e)
                        bad_data.append(ID)
    print('bad data:', bad_data)
    df = pd.DataFrame({'BCH MRN': MRNs, 'MRI Date': dates})
    df.drop_duplicates(subset=['BCH MRN'], keep='first', inplace=True, ignore_index=True)
    print('total MRI:', df.shape[0])
    df.to_csv(data_dir + '/MRI_date.csv')


def combine():
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data/clinical_data'
    df1 = pd.read_csv(proj_dir + '/BCH_MRI_clinical.csv')
    df2 = pd.read_csv(proj_dir + '/MRI_date.csv')
    print(df1.shape[0])
    print(df2.shape[0])
    df = df1.merge(df2, on='BCH MRN', how='left').reset_index()
    df.drop_duplicates(subset=['BCH MRN'], keep='first', inplace=True, ignore_index=True)
    df = df[['BCH MRN', 'DFCI MRN', 'MRI', 'MRI Date', 'LAST NAME', 'FIRST NAME', 'DOB', 'Date of Path Diagnosis']]
    df.to_csv(proj_dir + '/BCH_sum.csv')


def test():
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG/CBTN_BCH_Data/BCH/BCH_curated'
    data_dir = proj_dir + '/Abrecht_Alexandra_S_1187493/20010514/AX_FSE_T2/IM-0003-0001.dcm'
    ds = dicom.read_file(data_dir)
    print(ds)


def get_dcm_header():
    """ get all dicom header tags for all the patients
    """    
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data'
    MRNs = []
    dates = []
    bad_data = []
    IDs = []
    dfs = []
    count = 0
    df0 = pd.DataFrame(columns=['name'])
    for dataset in ['BCH_curated', 'BCH_curated2', 'BCH_curated3']:
        for root, dirs, files in os.walk(proj_dir + '/BCH_raw/' + dataset):
            if not dirs:
                if 'T2' in root:
                    #print(root)
                    count += 1
                    dcm_dirs = [i for i in glob.glob(root + '/*dcm')]
                    ds = dicom.read_file(dcm_dirs[0], force=True)
                    #print(ds)
                    ID = dcm_dirs[0].split('/')[-4]
                    print(count, ID)
                    print(ds.values())
                    df = pd.DataFrame(ds.values())
                    df[0] = df[0].apply(lambda x: dicom.dataelem.DataElement_from_raw(x)
                            if isinstance(x, dicom.dataelem.RawDataElement) else x)
                    df['name'] = df[0].apply(lambda x: x.name)
                    df['value'] = df[0].apply(lambda x: x.value)
                    df = df[['name', 'value']].set_index('name')
                    df.drop('Pixel Data', axis=0, inplace=True)
                    df.reset_index(inplace=True)
                    IDs.append(ID)
                    dfs.append(df)
                    #print(df)
    #pd.set_option("display.max_rows", 100, "display.max_columns", 100)
    df = pd.concat([df.set_index('name') for df in dfs], axis=1, join='inner').reset_index()
    df.columns = ['name'] + IDs
    df = df.set_index('name').T.reset_index()
    df.to_csv(proj_dir + '/clinical_data/BCH_dicom_header.csv', index=False)


def BRAF_TOT():
    """combine BRAF from redcap and pathology department
    """
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data'
    clinical_dir = proj_dir + '/clinical_data'
    data_dir = proj_dir + '/BCH_raw/BCH_TOT'
    df0 = pd.read_csv(clinical_dir + '/BCH_data_sum.csv')
    df1 = pd.read_csv(clinical_dir + '/10_417_BRAF_Cases_for_Dr_Kann.csv')
    df2 = pd.read_csv(clinical_dir + '/CharacterizationOfCl_DATA_LABELS_2022-11-21_0829.csv')
    df0['BCH MRN'] = df0['BCH MRN'].astype(int)
    df1['BCH MRN'] = df1['BCH MRN'].astype(int)
    df2['BCH MRN'] = df2['BCH MRN'].fillna(0).astype(int)
    IDs = []
    for folder in os.listdir(data_dir):
        if not folder.startswith('.'):
            ID = folder.split('_')[-1]
            print(ID)
            ID = int(ID)
            IDs.append(ID)
    MRIs = []
    for MRN in df0['BCH MRN'].to_list():
        if MRN in IDs:
            MRI = 'yes'
        else:
            MRI = 'no'
        MRIs.append(MRI)
    MRNs = []
    for MRN in df1['BCH MRN'].to_list():
        if MRN not in df0['BCH MRN'].to_list():
            print(MRN)
            MRNs.append(MRN)
    print(len(MRNs))
    #print(MRNs)

    df0['MRI data'] = MRIs
    df0 = df0[['BCH MRN', 'DFCI MRN', 'MRI data', 'LAST NAME', 'FIRST NAME', 'DOB', 'Date of Path Diagnosis']]
    df1 = df1[['BRAF V600E IHC Status', 'BRAF V600E Sequencing Status', 'BRAF Fusion Status', 'BCH MRN']]  
    df2 = df2[['BRAF V600E mutation', 'BRAF V600E Testing (choice=IHC)', 
        'BRAF V600E Testing (choice=Sequencing)', 'BRAF V600E Testing (choice=Other)', 
        'BRAF V600E Other Testing', 'BRAF fusion', 'BRAF fusion Information', 'BRAF fusion testing (choice=ISH)',
        'BRAF fusion testing (choice=Sequencing)', 'BRAF fusion testing (choice=Other)', 
        'Braf fusion testing other', 'BCH MRN']]
    df = df0.merge(df1, on='BCH MRN', how='left').reset_index()
    df = df.merge(df2, on='BCH MRN', how='left').reset_index()
    df.drop(['level_0', 'index'], axis=1, inplace=True)
    print(df)
    df.to_csv(clinical_dir + '/BRAF_TOT.csv', index=False)

    # BRAF more
    df1 = pd.read_csv(clinical_dir + '/10_417_BRAF_Cases_for_Dr_Kann.csv', index_col=0)
    df = df1.merge(df, on='BCH MRN', how='left').reset_index()
    df.to_csv(clinical_dir + '/BRAF_temp.csv', index=False)

    df = df[~df['MRI data'].isin(['yes', 'no'])]
    #df = df[df['OncoPanel'].isin(['Yes'])]
    df2 = df[df['WHO Classification Grade'].isin(['I', 'II', 'I-II'])]
    df2 = df2[df2['BRAF V600E Sequencing Status_x'].isin(['Positive', 'Positive (p.G596R)', 'Negative'])]

    df3 = df[df['BRAF V600E Sequencing Status_x'].isin(['Positive', 'Positive (p.G596R)'])]
    df3 = df3[~df3['WHO Classification Grade'].isin(['I', 'I-II', 'II'])]
    df = pd.concat([df2, df3])
    df = df[['BCH MRN', 'MRI data', 'Date of Diagnosis (MRI)', 'Last Name', 'First Name', 'Date of Birth', 
        'OncoPanel', 'WHO Classification Grade', 'BRAF V600E Sequencing Status_x', 'BRAF Fusion Status_x']]
    print(df)
    df.to_csv(clinical_dir + '/BRAF_more.csv', index=False)


def BRAF_more():
    """combine BRAF from redcap and pathology department
    """
    proj_dir = '/mnt/kannlab_rfa/Zezhong/pLGG/data'
    clinical_dir = proj_dir + '/clinical_data'
    # BRAF more
    df0 = pd.read_csv(clinical_dir + '/CharacterizationOfCl_DATA_LABELS_2022-11-21_0829.csv')
    redcap_list = df0['BCH MRN'].to_list()
    # exclude patients from redcap database
    df1 = pd.read_csv(clinical_dir + '/10_417_BRAF_Cases_for_Dr_Kann.csv', index_col=0)
    df = df1[~df1['BCH MRN'].isin(redcap_list)]
    df.to_csv(clinical_dir + '/BRAF_temp2.csv', index=False)
    
    # create BRAF any combining sequencing and IHC data
    brafs = []
    for ihc, seq in zip(df['BRAF V600E IHC Status'], df['BRAF V600E Sequencing Status']):
        if seq == 'Positive' or seq == 'Positive (p.G596R)':
            braf = 'Positive'
            brafs.append(braf)
        elif seq == 'Negative':
            braf = 'Negative'
            brafs.append(braf)
        else:
            if ihc == 'Positive' or ihc == 'Neagtive':
                braf = ihc
                brafs.append(braf)
            else:
                braf = 0
                brafs.append(braf)
    print(brafs)
    df['BRAF Any'] = brafs

    # WHO LGG with BRAF status
    df2 = df[df['WHO Classification Grade'].isin(['I', 'II', 'I-II'])]
    df2 = df2[df2['BRAF Any'].isin(['Positive', 'Negative'])]
    
    # WHO HGG with BRAF V600E
    df3 = df[df['BRAF Any'].isin(['Positive'])]
    df3 = df3[~df3['WHO Classification Grade'].isin(['I', 'I-II', 'II'])]

    # with no BRAF V600E status but with BRAF fusion status 
    df4 = df[df['BRAF Any']==0]
    print(df4)
    df4 = df4[df4['BRAF Fusion Status'].isin(['Positive', 'Negative'])]
    print(df4)

    df = pd.concat([df2, df3, df4])
    #print(df)
    df.to_csv(clinical_dir + '/BRAF_more2.csv', index=False)


if __name__ == '__main__':

    #get_clinical()
    #get_dcm_header()
    #combine()
    #test()
    BRAF_more()


