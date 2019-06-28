from keras.models import Model
from keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras import backend as K

# Design our model architecture
def create_model(img_hw, input_channels, dropout_value):
    #n_ch_exps = [4, 5, 6, 7, 8, 9]
    n_ch_exps = [4,5,6,7]
    #n_ch_exps = [6,7,8,9,10]
    #n_ch_exps = [6,7,8]
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (input_channels, img_hw, img_hw)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_hw, img_hw, input_channels)


    inp = Input(shape=input_shape)
    lambd = Lambda(lambda x: x / 255) (inp)

    encodeds = []
    # encoder
    enc = lambd
    #print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(enc)
        end = BatchNormalization()(enc)
        end = Activation('relu')(enc)

        enc = Dropout(dropout_value)(enc)
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(enc)
        end = BatchNormalization()(enc)
        end = Activation('relu')(enc)

        encodeds.append(enc)
        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)
    
    # decoder
    dec = enc
    #print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    #print(decoder_n_chs)
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

        dec = Dropout(dropout_value)(dec)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])
    
    return model

def create_model_atrous(img_hw, input_channels, dropout_value):
    use_bias=False
    #n_ch_exps = [4, 5, 6, 7, 8, 9]
    n_ch_exps = [5,6,7]
    #n_ch_exps = [6,7,8,9,10]
    #n_ch_exps = [6,7,8]
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (input_channels, img_hw, img_hw)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_hw, img_hw, input_channels)


    inp = Input(shape=input_shape)
    lambd = Lambda(lambda x: x / 255) (inp)

    encodeds = []
    # encoder
    enc = lambd
    #print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init, use_bias=use_bias)(enc)
        end = BatchNormalization()(enc)
        end = Activation('relu')(enc)

        enc = Dropout(dropout_value)(enc)
        #enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init, use_bias=use_bias)(enc)
        end = BatchNormalization()(enc)
        end = Activation('relu')(enc)

        encodeds.append(enc)
        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)

    # branching atrous layer
    n_ch = 2**(n_ch_exps[-1]+1)
    # 4
    b1 = Conv2D(filters=n_ch, kernel_size=k_size, padding='same', dilation_rate=(4, 4), use_bias=use_bias, kernel_initializer=k_init)(enc)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = Dropout(dropout_value)(b1)
    b1 = Conv2D(filters=n_ch, kernel_size=(1, 1), padding='same', use_bias=use_bias, kernel_initializer=k_init)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    # 8
    b2 = Conv2D(filters=n_ch, kernel_size=k_size, padding='same', dilation_rate=(8, 8), use_bias=use_bias, kernel_initializer=k_init)(enc)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = Dropout(dropout_value)(b2)
    b2 = Conv2D(filters=n_ch, kernel_size=(1, 1), padding='same', use_bias=use_bias, kernel_initializer=k_init)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    # 12
    b3 = Conv2D(filters=n_ch, kernel_size=k_size, padding='same', dilation_rate=(12, 12), use_bias=use_bias, kernel_initializer=k_init)(enc)
    b3 = BatchNormalization()(b3)
    b3 = Activation('relu')(b3)
    b3 = Dropout(dropout_value)(b3)
    b3 = Conv2D(filters=n_ch, kernel_size=(1, 1), padding='same', use_bias=use_bias, kernel_initializer=k_init)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation('relu')(b3)

    # merge
    bout = concatenate([b1, b2, b3], axis=-1)

    # decoder
    dec = bout
    #print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    #print(decoder_n_chs)
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init, use_bias=use_bias)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

        dec = Dropout(dropout_value)(dec)
        #dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, padding='same', kernel_initializer=k_init, use_bias=use_bias)(dec)
        dec = BatchNormalization()(dec)
        dec = Activation('relu')(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])

    return model

if __name__ == "__main__":
    model = create_model(img_hw=96, input_channels=3, dropout_value=0.5)
    model.summary()
    from keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model.png')
