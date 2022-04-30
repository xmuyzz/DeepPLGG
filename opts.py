import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    
    # path
    parser.add_argument('--root_dir', default='/mnt/aertslab/USERS/Zezhong/pLGG', type=str, help='Root path')
    parser.add_argument('--curation', default='curation', type=str, help='Data curation path')
    parser.add_argument('--out', default='out', type=str, help='Results output path')
    parser.add_argument('--pro_data', default='pro_data', type=str, help='Processed data path')
    parser.add_argument('--model', default='model', type=str, help='Models output path')
    parser.add_argument('--log', default='log', type=str, help='Log data path')
    parser.add_argument('--nii', default='nii', type=str, help='nii path')
    
    # data preprocessing
    #parser.add_argument('--BRAF', default='BRAF_fusion', type=str, help='BRAF (BRAF_status|BRAF_fusion)')
    parser.add_argument('--manual_seed', default=1234, type=int, help='seed')
    parser.add_argument('--channel', default=3, type=int, help='Input channel (3 | 1)')
    
    # train model
    parser.add_argument('--task', default='BRAF_status', type=str, help='BRAF|tumor')
    parser.add_argument('--run_type', default='test', type=str, help='train|val|test')
    parser.add_argument('--weights', default='imagenet', type=str, help='None|imagenet')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epoch', default=10, type=int, help='Epoch')
    parser.add_argument('--activation', default='sigmoid', type=str, help='Activation function on last layer')
    parser.add_argument('--loss_function',  default='binary_crossentropy', type=str, help='loss function')
    parser.add_argument('--cnn_model', default='simple_cnn', type=str, help='cnn model')
    parser.add_argument('--input_shape', default=(192, 192, 3), type=int, help='Input shape')
    parser.add_argument('--freeze_layer', default=None, type=str, help='Freeze layer to train')
    parser.add_argument('--trained_weights', default='10-0.89.h5', type=str, help='Model weights for fine tuning')
    
    # test model   
    parser.add_argument('--thr_img', default=0.5, type=float, help='threshold to decide class on image level')
    parser.add_argument('--thr_prob', default=0.5, type=float, help='threshold to decide class on patient level')
    parser.add_argument('--thr_pos', default=0.5, type=float, help='threshold to decide class on patient level')
    parser.add_argument('--n_bootstrap', default=50, type=int, help='n times of bootstrap to calcualte 95% CI')
    parser.add_argument('--saved_model', default='04-0.91.h5', type=str, help='saved model name')    
    parser.add_argument('--_load_model', default='load_weights', type=str, help='load_model|load_weights')
    # fine tune model
    parser.add_argument('--tuned_model', default='Tuned_EffNetB4', type=str, help='tuned model')    
    
    # actions
    parser.add_argument('--load_data', action='store_true', help='If true, load data is performed.')
    parser.set_defaults(load_data=True)
    parser.add_argument('--save_nii', action='store_true', help='If true, load data is performed.')
    parser.set_defaults(save_nii=False)
    parser.add_argument('--train', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train=False)
    parser.add_argument('--test', action='store_true', help='If true, validation is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--stats_plots', action='store_true', help='If true, plots and statistics is performed.')
    parser.set_defaults(stats_plots=True)

    args = parser.parse_args()

    return args







