import os
import numpy as np
import pandas as pd



def MRI_curation(data_dir, proj_dir):

    """ MRI data curation
    """

    curation_dir = os.path.join(proj_dir, 'curation')
    if not os.path.exists(curation_dir): os.mkdir(curation_dir)

    dirs = []
    pat_ids = []
    scan_ids = []
    seq_ids = []
    file_ids = []
    views = []
    contrasts = []
    scan_types = []
    scan_dates = []

    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            ## find nii files
            f_format = name.split('/')[-1].split('.')[-2].strip()
            if f_format == 'nii':
                file_ids.append(name)
                dir = os.path.join(path, name)
                dirs.append(dir)
                
                ## patient ID
                pat_id = dir.split('/')[9].strip()
                pat_ids.append(pat_id)
                
                ## scan ID
                scan_id = dir.split('/')[11].strip()
                scan_ids.append(scan_id)
                
                # scan type: brain or spine
                scan_type = scan_id.split('_')[2]
                scan_types.append(scan_type)
                
                ## scan date
                scan_date = scan_id.split('_')[0][:-1]
                scan_dates.append(scan_date)
                
                ## MRI sequence types
                #---------------------
                fs = name.split('_')
                fs[-1] = fs[-1].split('.')[0].strip() 
                #print(fs)
                seq_set = ['T1', 't1', 'T2', 't2', 'flair', 
                            'FLAIR', 'dark', 'ADC', 'FA']
                overlap = list(set(seq_set) & set(fs)) 
                if not overlap:
                    seq_id = fs[1]
                else:
                    ## flair will have t2 and flair
                    if len(overlap) == 2 and list(set(overlap) & set(['t2', 'T2'])):
                        #print(overlap)
                        seq_id = 'FLAIR'
                    else:
                        seq_id = overlap[0]
                seq_ids.append(seq_id)
                
                ## choose scan view
                #-------------------
                view_list = ['ax', 'AX', 'Ax', 'axial', 'AXIAL', 'tra', 'TRA', 
                             'trans', 'sag', 'SAG', 'Sag', 'cor', 'COR', 'Cor']
                overlap = list(set(view_list) & set(fs))
                if not overlap:
                    view = ' '
                else:
                    view = overlap[0]
                views.append(view)
    
                # T1W contrast
                #-----------------
                contrast_list = ['pre', 'PRE', 'Pre', 'post', 'POST', 'Post']
                overlap = list(set(contrast_list) & set(fs))
                if not overlap:
                    contrast = ' '
                else:
                    contrast = overlap[0]
                contrasts.append(contrast)
    
    ## rename sequences    
    #-------------------
    seq_ids = ['T1W' if i in ['t1', 'T1'] 
                else 'T2W' if i in ['t2', 'T2'] 
                else i for i in seq_ids] 
    
    ## rename views     
    #----------------
    views = ['tra' if i in ['ax', 'AX', 'Ax', 'axial', 'AXIAL', 'tra', 'trans', 'TRA']
             else 'sag' if i in ['sag', 'SAG', 'Sag']
             else 'cor' if i in ['cor', 'COR', 'Cor']
             else i for i in views]
    # ADC and FA maps are also tra view
    scan_views = []
    for seq, view in zip(seq_ids, views):
        if seq in ['ADC', 'FA']:
            view = 'tra'
        else:
            view = view
        scan_views.append(view)
            
    ## rename contrasts 
    #------------------
    contrasts = ['pre' if i in ['pre', 'Pre', 'PRE']
                 else 'post' if i in ['post', 'POST', 'Post']
                 else i for i in contrasts]

    df = pd.DataFrame({'pat_id': pat_ids,
                       'scan_id': scan_ids,
                       'scan_date': scan_dates,
                       'scan_type': scan_types,
                       'seq_id': seq_ids,
                       'view': scan_views,
                       'contrast': contrasts,
                       'file_id': file_ids,
                       'path': dirs})
    
    df.to_csv(os.path.join(curation_dir, 'MRI_master.csv'), index=False)
    print('MRI data curaiton complete!!')


def df_filter(proj_dir):
    
    """ filter df
    """

    curation_dir = os.path.join(proj_dir, 'curation')
    if not os.path.exists(curation_dir): os.mkdir(curation_dir)
    
    df = pd.read_csv(os.path.join(curation_dir, 'MRI_master.csv'))
    
    ## keep the pre operation scan using the min date
    #-----------------------------------------------
    min_date = df.groupby(['pat_id']).scan_date.min().reset_index(name='min_date')
    df = df.merge(min_date, how='left', on=['pat_id'])
    df0 = df[df['scan_date']==df['min_date']]
    
    ## choose scan type: brain, spine, orbit, pituitary
    #---------------------------------------------------
    df1 = df0[(df0['scan_type'].isin(['brain'])) & (df0['view'].isin(['tra']))]
    # find patient without brain scan
    scan_types = []
    pat_ids = []
    for scan_type, pat_id in zip(df0['scan_type'], df0['pat_id']):
        if scan_type != 'brain' and pat_id not in df1['pat_id']:
            scan_types.append(scan_type)
            pat_ids.append(pat_id)
    df2 = pd.DataFrame({'pat_id': pat_ids, 'scan_type': scan_types})
    df2.drop_duplicates('pat_id', keep='first', inplace=True)
    print(df2)
    print('total patients without brain scan:', df2.shape[0])
    
    # save df to csv
    #----------------
    df3 = pd.concat([df1, df2], axis=0, ignore_index=True)
    df3.to_csv(os.path.join(curation_dir, 'curation_data.csv'), index=False)
    print('data curation complete!!')

            
