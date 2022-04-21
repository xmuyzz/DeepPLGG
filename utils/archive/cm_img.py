from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix





def cm_img(run_type, save_dir):
    
    ### determine if this is train or test
    if run_type == 'val':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_val_pred.p'))
    if run_type == 'test':
        df_sum = pd.read_pickle(os.path.join(save_dir, 'df_test_pred.p')

    y_true = df_sum['label'].to_numpy()
    y_pred = df_sum['y_pred'].to_numpy()
    
    ### Create confusion matrix
    cm = confusion_matrix(y_test, y_pred_class)
    cm_norm = cm.astype('float64') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 3)

    ## plot cm
    for cm, cm_type in zip([cm, cm_norm], ['raw', 'norm']):
        plot_cm(
            cm=cm,
            cm_type='raw',
            level='patient',
            save_dir=save_dir
            )
    
    ### classification report
    report = classification_report(y_test, y_pred_class, digits=3)
    
    print('confusion matrix:')
    print(cm)
    print(cm_norm)
    print(report)

    return cm, cm_norm, report
