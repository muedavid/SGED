from tensorflow.keras import layers


# TODO don't import whole

def convolution_block(block_input, num_filters=24, kernel_size=3, dilation_rate=1, padding="same", use_bias=False,
                      seperable=False, BN=True, RELU=True, name='conv_block'):
    if seperable:
        x = layers.SeparableConv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                                   use_bias=use_bias,name=name+'_separable_conv')(block_input)
    else:
        x = layers.Conv2D(num_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding=padding,
                          use_bias=use_bias, name=name+'_conv')(block_input)
    if BN:
        x = layers.BatchNormalization(name=name+'_bn')(x)
    if RELU:
        x = layers.ReLU(name=name+'_relu')(x)
    return x


# TODO: DASPP if not good working: test if BN and RELU needed
def DASPP(dspp_input):
    dims = dspp_input.shape

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1, BN=True, RELU=True)
    out_3 = convolution_block(dspp_input, kernel_size=3, dilation_rate=3, seperable=True, BN=False, RELU=False)
    out_3 = convolution_block(out_3, kernel_size=3, dilation_rate=1, seperable=True, BN=True, RELU=True)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6, seperable=True, BN=False, RELU=False)
    out_6 = convolution_block(out_6, kernel_size=3, dilation_rate=1, seperable=True, BN=True, RELU=True)
    out_9 = convolution_block(dspp_input, kernel_size=3, dilation_rate=9, seperable=True, BN=False, RELU=False)
    out_9 = convolution_block(out_9, kernel_size=3, dilation_rate=1, seperable=True, BN=True, RELU=True)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18, seperable=True, BN=False, RELU=False)
    out_18 = convolution_block(out_18, kernel_size=3, dilation_rate=1, seperable=True, BN=True, RELU=True)

    out = layers.GlobalAveragePooling2D(keepdims=True)(dspp_input)
    out = convolution_block(out, kernel_size=1, dilation_rate=1, BN=True, RELU=True)

    out_pool = layers.UpSampling2D(size=(dims[-3], dims[-2]), interpolation="bilinear")(out)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_3, out_6, out_9, out_18, dspp_input])

    x = convolution_block(x, kernel_size=1)

    return x


def side_feature_casenet(x, channels, kernel_size_transpose, stride_transpose, output_padding=None, padding='same',
                         name=None):
    x = layers.Conv2D(channels, kernel_size=1, strides=(1, 1), padding='same')(x)
    return layers.Conv2DTranspose(channels, kernel_size=kernel_size_transpose,
                                  strides=(stride_transpose, stride_transpose), padding=padding,
                                  output_padding=output_padding, use_bias=False, name=name)(x)


# TODO: DASPP if not good working: test if BN and RELU needed in second conv.
def side_feature_SGED(x, output_dims, interpolation="bilinear"):
    x = convolution_block(x, kernel_size=3, dilation_rate=1, BN=True, RELU=True)
    x = convolution_block(x, num_filters=1, kernel_size=1, dilation_rate=1, BN=True, RELU=False)

    if x.shape[1] != output_dims[0]:
        x = layers.UpSampling2D(size=(output_dims[0]//x.shape[1], output_dims[1]//x.shape[2]), interpolation=interpolation)(x)
    return x


def decoder(daspp, side, output_dims, NUM_CLASSES=3, num_side_filters=6):
    x = layers.UpSampling2D(size=(side.shape[1]//daspp.shape[1], side.shape[2]//daspp.shape[2]), interpolation="bilinear")(daspp)

    side = convolution_block(side, num_filters=num_side_filters, kernel_size=1, dilation_rate=1, seperable=False,
                             BN=True, RELU=True)

    x = layers.Concatenate(axis=-1)([x, side])

    x = convolution_block(x, kernel_size=3, dilation_rate=1, seperable=True, BN=True, RELU=True)

    x = layers.UpSampling2D(size=(output_dims[0]//x.shape[1], output_dims[1]//x.shape[2]), interpolation="bilinear")(x)

    x = convolution_block(x, kernel_size=3, dilation_rate=1, seperable=True, BN=True, RELU=True)

    x = convolution_block(x, num_filters=NUM_CLASSES, kernel_size=1, dilation_rate=1, seperable=False, BN=True,
                          RELU=True)

    return x


def shared_concatenation(sides, num_classes):
    shared_concat = []
    for i in range(num_classes):
        for j in range(len(sides)):
            if sides[j].shape[-1] == num_classes:
                shared_concat.append(sides[j][:, :, :, i:i + 1])
            else:
                shared_concat.append(sides[j])

    return layers.Concatenate(axis=-1)(shared_concat)


def fused_classification(x, num_classes, name=None):
    return layers.Conv2D(filters=num_classes, kernel_size=1, groups=num_classes, name=name)(x)
