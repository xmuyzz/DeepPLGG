from data_curation.MRI_curation import MRI_curation
from data_curation.MRI_curation import df_filter
from data_curation.BRAF_curation import BRAF_curation
from data_curation.MRI_Sequences import MRI_Sequences


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/Glioma/flywheel_20210210_223349/flywheel/LGG/SUBJECTS'
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    run_MRI_curation = False
    
    if run_MRI_curation:
        MRI_curation(
            data_dir=data_dir,
            proj_dir=proj_dir
            )

    df_filter(proj_dir=proj_dir)

    MRI_Sequences(proj_dir=proj_dir)

    BRAF_curation(proj_dir=proj_dir)
