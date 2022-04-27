import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import (
    ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
    EfficientNetB5, EfficientNetB6, EfficientNetB7, MobileNet, MobileNetV2, VGG16, VGG19,
    InceptionV3, Xception, InceptionResNetV2, DenseNet121, DenseNet169, DenseNet201)
from models.simple_cnn import simple_cnn



def generate_model(cnn_model, weights, freeze_layer, input_shape, activation, loss_function, lr):


    """
    EfficientNet

    Args:
      effnet       - required : name of resnets with different layers, i.e. 'ResNet101'.
      transfer     - required : decide if want do transfer learning
      model_dir    - required : folder path to save model
      freeze_layer - required : number of layers to freeze
      activation   - required : activation function in last layer
    
    """
       
    if cnn_model == 'simple_cnn':
        model = simple_cnn(input_shape=input_shape, activation=activation)
    else:
        ### determine input shape
        default_shape = (224, 224, 3)
        if input_shape == default_shape:
            include_top = True
        else:
            include_top = False

        # Inception
        if cnn_model == 'InceptionV3':
            base_model = InceptionV3(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'Xception':
            base_model = Xception(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'InceptionResNetV2':
            base_model = InceptionResNetV2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # ResNet
        elif cnn_model == 'ResNet50V2':
            base_model = ResNet50V2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet101V2':
            base_model = ResNet101V2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet152V2':
            base_model = ResNet152V2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        if cnn_model == 'ResNet50':
            base_model = ResNet50(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet101':
            base_model = ResNet101(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'ResNet152':
            base_model = ResNet152(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # EfficientNet
        elif cnn_model == 'EfficientNetB0':
            base_model = EfficientNetB0(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB1':
            base_model = EfficientNetB1(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB2':
            base_model = EfficientNetB2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        if cnn_model == 'EfficientNetB3':
            base_model = EfficientNetB3(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB4':
            base_model = EfficientNetB4(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EffcientNetB5':
            base_model = EfficientNetB5(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EfficientNetB6':
            base_model = EfficientNetB6(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'EffcientNetB7':
            base_model = EfficientNetB7(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # MobileNet
        if cnn_model == 'MobileNet':
            base_model = MobileNet(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'MobileNetV2':
            base_model = MobileNetV2(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # VGG
        if cnn_model == 'VGG16':
            base_model = VGG16(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'VGG19':
            base_model = VGG19(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)

        # DenseNet
        if cnn_model == 'DenseNet121':
            base_model = DenseNet121(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'DenseNet169':
            base_model = VGG19(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        elif cnn_model == 'DenseNet201':
            base_model = DenseNet201(weights=weights, include_top=include_top,
                input_shape=input_shape, pooling=None)
        
        base_model.trainable = True

     ### create top model
        inputs = base_model.input
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(1000, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(1, activation=activation)(x)
        model = Model(inputs=inputs, outputs=outputs)

        ### freeze specific number of layers
        if freeze_layer == 1:
            for layer in base_model.layers[0:5]:
                layer.trainable = False
            for layer in base_model.layers:
                print(layer, layer.trainable)
        if freeze_layer == 5:
            for layer in base_model.layers[0:16]:
                layer.trainable = False
            for layer in base_model.layers:
                print(layer, layer.trainable)
        else:
            for layer in base_model.layers:
                layer.trainable = True
        model.summary()

#    # model compile
#    auc = tf.keras.metrics.AUC()
#    model.compile(
#        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#        loss=loss_function,
#        metrics=[auc])

    return model




