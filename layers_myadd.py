import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers as L
import tensorflow_addons as tfa

class SqueezeExciteLayer(L.Layer):
    def __init__(self, ratio,*args,**kwagrs):
        super().__init__(*args,**kwagrs)
        self.ratio = ratio
        self.pool = None
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):
       orig_size = input_shape[-1]
       squeeze_size = max(orig_size // self.ratio , 4 )
       self.pool = L.GlobalAveragePooling2D()
       self.dense1 = L.Dense(squeeze_size,activation='relu')
       self.dense2 = L.Dense(orig_size,activation='sigmoid')

    def call(self,batch_input):
        x = self.pool(batch_input)
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.reshape(x, shape=(-1, 1, 1, batch_input.shape[-1]))
        return x * batch_input


class ResidualDecoderCell(L.Layer):
    def __init__(self,
                 expand_ratio=6,
                 se_ratio=16,
                 bn_momentum=0.95,
                 gamma_reg=None,
                 use_bias=True,
                 res_scalar=0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.gamma_reg = gamma_reg
        self.use_bias = use_bias
        self.res_scalar = res_scalar
        
        self.conv_depthw = None
        self.conv_expand = None
        self.conv_reduce = None

    def build(self,input_shape):
   
        self.conv_3x3s_0 = NvaeConv2D(kernel_size=(3, 3)) #分离卷积效果不好，使用简单的conv2d效果最好
        self.conv_5x5s_1 = NvaeConv2D(kernel_size=(3, 3))   
        self.se_layer = SqueezeExciteLayer(self.se_ratio)

    def call(self,batch_input,training = None):
        x = batch_input
        x = self.conv_3x3s_0(x,training = training)
        x = tf.keras.activations.swish(x)
        x = self.conv_5x5s_1(x,training = training)
        x = tf.keras.activations.swish(x)
        x = self.se_layer(x, training=training)

        residual = batch_input
        
        output = L.Add()([self.res_scalar * residual, x])

        return output

class ResidualEncoderCell(L.Layer):
    def __init__(self,
                 se_ratio=16,
                 bn_momentum=0.95,
                 gamma_reg=None,
                 use_bias=True,
                 res_scalar=0.1,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.se_ratio = se_ratio
        self.bn_momentum = bn_momentum
        self.gamma_reg = gamma_reg
        self.use_bias = use_bias
        self.res_scalar = res_scalar
        self.conv_3x3s_0 = None
        self.conv_3x3s_1 = None
        self.se_layer = None


    def build(self, input_shape):      
        self.conv_3x3s_0 = NvaeConv2D(kernel_size=(3, 3))
        self.conv_5x5s_1 = NvaeConv2D(kernel_size=(3, 3))
        self.se_layer = SqueezeExciteLayer(self.se_ratio)

    def call(self, batch_input, training=None):
        x = batch_input
        x = self.conv_3x3s_0(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.conv_5x5s_1(x, training=training)
        x = tf.keras.activations.swish(x)
        x = self.se_layer(x)

        residual = batch_input
            
        output = L.Add()([self.res_scalar* residual,  x])
        
        return output
    


class NvaeConv2D(L.Layer):
    def __init__(self,
                 kernel_size,
                 scale_channels=1,
                 depthwise=False,
                 use_bias=True,
                 weight_norm=False,
                 spectral_norm=True,
                 dilation_rate=1,
                 activation='linear',
                 padding='same',
                 abs_channels=None,
                 *args,
                 **kwargs):
        super().__init__(*args,**kwargs)
        self.abs_channels = abs_channels
        self.scale_channels = scale_channels
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.use_bias = use_bias
        self.weight_norm = weight_norm
        self.spectral_norm = spectral_norm
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.padding = padding
        self.channels_in = None        
        self.conv = None
        self.conv_depth1x1 = None

    def build(self, input_shape):
        self.channels_in = input_shape[-1]
        if self.abs_channels is None: #-1表示upsampling,abs_channel是最终输出的维度s
            assert self.scale_channels != 0
            if self.scale_channels > 0 :
                self.abs_channels = self.channels_in * self.scale_channels
            else :
                self.abs_channels = self.channels_in // abs(self.scale_channels)
        self.conv = L.Conv2D(
            filters = self.abs_channels,
            kernel_size = self.kernel_size,
            strides = 1,
            groups = 1 if not self.depthwise else self.abs_channels,##分离卷积
            use_bias = self.use_bias,
            dilation_rate = self.dilation_rate,
            activation = self.activation,
            padding =self.padding,
        )
        self.conv_depth1x1 = None

        if self.depthwise:
            self.conv_depth1x1 = L.Conv2D(self.abs_channels, kernel_size=(1, 1))
        

    def call(self,x,training = False):
        x = self.conv(x, training=training)

        if self.depthwise:
            x = self.conv_depth1x1(x, training=training)
        return x
