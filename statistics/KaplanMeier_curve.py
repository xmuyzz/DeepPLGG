import os
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from time import localtime, strftime
from sklearn.cluster import KMeans



def main(proj_dir, threshold, n_clusters):

    """
    Kaplan-Meier analysis for risk group stratification

    Args:
        proj_dir {path} -- project dir;
        out_dir {path} -- output dir;
        score_type {str} -- prob scores: mean, median, 3-year survival;
        hpv {str} -- hpv status: 'pos', 'neg';
    
    Returns:
        KM plot, median survial time, log-rank test;
    
    Raise errors:
        None;

    """

    pred = pd.read_csv(os.path.join(pro_data_dir, 'test_pred_2yr.csv'), index_col=0)
    pred.columns = ['Subject_ID', 'Slice_ID', 'label', 'y_pred', 'y_pred_class']
    pred = pred.groupby(['Subject_ID']).mean().reset_index()
    df = pd.read_csv(os.path.join(curation_dir, 'BRAF_survival_slice.csv'))
    df = pred.merge(df, how='left', on='Subject_ID')
    
    # get median scores to stratify risk groups
    groups = []
    if threshold == 'median':
        for y_pred in df['y_pred']:
            if y_pred > df['y_pred'].median():
                group = 1
            else:
                group = 0
            groups.append(group)
        print(groups)
    elif threshold == 'kmeans':
        k_means = KMeans(
            n_clusters=n_clusters,
            algorithm='auto', 
            copy_x=True, 
            init='k-means++', 
            max_iter=300,
            random_state=0, 
            tol=0.0001, 
            verbose=0)
        prob_scores = df['y_pred'].to_numpy().reshape(-1, 1)
        k_means.fit(prob_scores)
        groups = k_means.predict(prob_scores)
        print(groups)
    df['group'] = groups
    #print(df)

    # multivariate log-rank test
    results = multivariate_logrank_test(
        df['PFS'],
        df['group'],
        df['2yr_event'])
    #results.print_summary()

    #p_value = np.around(results.p_value, 3)
    print('log-rank test p-value:', results.p_value)
    #print(results.test_statistic)

    # Kaplan-Meier curve
    df1 = df.loc[df['group'] == 1]
    PFS1 = 1 - df1.loc[df['2yr_event'] == 1].shape[0]/df1.shape[0]
    PFS1 = round(PFS1, 3)
    print('PFS1:', PFS1)
    print('group 1:', df1.shape[0])
    df0 = df.loc[df['group'] == 0]
    PFS0 = 1 - df0.loc[df['2yr_event'] == 1].shape[0]/df0.shape[0]
    FPS0 = round(PFS0, 3)
    print('PFS0:', PFS0)
    print('group 0:', df0.shape[0])
    dfs = [df0, df1]
    
#    dfs = []
#    for i in range(2):
#        df = df.loc[df['group'] == i]
#        print('df:', i, df)
#        dfs.append(df)
    
    # Kaplan-Meier plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    labels = ['High risk', 'Low risk']
    for df, label in zip(dfs, labels):
        #print(df)
        #print(df['PFS'])
        #print(df['2yr_event'])
        kmf = KaplanMeierFitter()
        kmf.fit(
            df['PFS'],
            df['2yr_event'],
            label=label)
        ax = kmf.plot_survival_function(
            ax=ax,
            show_censors=True,
            ci_show=True,
            #censor_style={"marker": "o", "ms": 60})
            )
        #add_at_risk_counts(kmf, ax=ax)
        median_surv = kmf.median_survival_time_
        median_surv_CI = median_survival_times(kmf.confidence_interval_)
        print('median survival time:', median_surv)
        print('median survival time 95% CI:\n', median_surv_CI)
    
    plt.xlabel('Time (days)', fontweight='bold', fontsize=12)
    plt.ylabel('Survival probability', fontweight='bold', fontsize=12)
    plt.xlim([0, 1000])
    plt.ylim([0, 1])
    #ax.patch.set_facecolor('gray')
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=1, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=5000, color='k', linewidth=2)
    plt.xticks([0, 200, 400, 600, 800, 1000], fontsize=12, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12, fontweight='bold')
    plt.legend(loc='lower left', prop={'size': 12, 'weight': 'bold'})
    #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower left', mode="expand", 
    #           borderaxespad=0, ncol=3, prop={'size': 12, 'weight': 'bold'})
    plt.grid(True)
    plt.title('Kaplan-Meier Survival Estimate', fontsize=16, fontweight='bold')
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    kmf_fn = '2yr_FPS.png'
    plt.savefig(os.path.join(output_dir, kmf_fn), format='png', dpi=300)
    plt.close()
    
    #print('saved Kaplan-Meier curve!')

    
if __name__ == '__main__':
    
    proj_dir = '/mnt/aertslab/USERS/Zezhong/pLGG'
    curation_dir = os.path.join(proj_dir, 'curation')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    output_dir = os.path.join(proj_dir, 'output')    
    
    main(proj_dir, threshold='kmeans', n_clusters=2)



