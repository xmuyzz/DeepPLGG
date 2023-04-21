from utils import gradcam
import numpy as np
import imutils
import cv2
import tensorflow


def plot_gradcam(input_channel, i):
    
    ### with input channel to derermine dataset
    if input_channel == 1:
        x_train = np.load(os.path.join(train_img_dir, 'train_arr.npy'))
        x_val = np.load(os.path.join(val_img_dir, 'val_arr.npy'))
    elif input_channel == 3:
        x_train = np.load(os.path.join(train_img_dir, 'train_arr_3ch.npy'))
        x_val = np.load(os.path.joub(val_img_dir, 'val_arr_3ch.npy')
    ### load dataset
    train_df = pd.read_pickle(os.path.join(train_img_dir, 'train_df.p'))
    val_df = pd.read_pickle(os.path.join(val_img_dir, 'val_df.p'))
    y_train = train_df['label']
    y_val = val_df['label']
    ### find the ith image to show grad-cam map
    image = x_train[i]
    label = y_train[i]

    ### load saved model
    model = load_model(os.path.join(output_dir, 'model'))
    pred = model.predict(image)
    classIdx = pred
    #classIdx = np.argmax(preds[0])

    for idx in range(len(model.layers)):
      print(model.get_layer(index = idx).name)

    ### we picked 'block5c_project_con' layer form model summary
    ### calculate gradient
    icam = GradCAM(model, i, 'conv5_block3_3_conv') 
    heatmap = icam.compute_heatmap(image)
#    heatmap = cv2.resize(heatmap, (64, 64))
#    image = cv2.imread('/content/dog.jpg')
#    image = cv2.resize(image, (32, 32))
    print(heatmap.shape, image.shape)
    ### generate overlaid grad_cam map
    (heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)

    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(heatmap)
    ax[1].imshow(image)
    ax[2].imshow(output)

   return output




