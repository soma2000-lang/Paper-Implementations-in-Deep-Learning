import tensorflow as tf

class Conv2d_Bn(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding='same', strides=1):
        super(Conv2d_Bn, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size,
                                           padding=padding, strides=strides)
        self.bn = tf.keras.layers.BatchNormalization(axis=3)
        self.act = tf.keras.layers.ReLU()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.act(x)
        return x


class inception_block(tf.keras.layers.Layer):
    def __init__(self):
        super(inception_block, self).__init__()
        self.conv1 = Conv2d_Bn(32, (3, 3), strides=2,padding='same')
        self.conv2 = Conv2d_Bn(32, (3, 3),padding='same')
        self.conv3 = Conv2d_Bn(64, (3, 3),padding='same')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='same')
        self.conv4 = Conv2d_Bn(80,(1,1),strides=1,padding='same')
        self.conv5 = Conv2d_Bn(192,(3,3),strides=1,padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='same')

    def call(self,inputs,training=None,**kwargs):
        x = self.conv1(inputs,training=training)
        x = self.conv2(x,training=training)
        x = self.conv3(x,training=training)
        x = self.pool1(x)
        x = self.conv4(x,training=training)
        x = self.conv5(x,training=training)
        x = self.pool2(x)
        return x


class inception_A(tf.keras.layers.Layer):
    def __init__(self):
        super(inception_A, self).__init__()
        self.conv0_x0 = Conv2d_Bn(64,(1,1))

        self.conv1_x1 = Conv2d_Bn(48,(1,1))
        self.conv1_x2 = Conv2d_Bn(64,(5,5))

        self.conv2_x1 = Conv2d_Bn(64, (1, 1))
        self.conv2_x2 = Conv2d_Bn(96, (3, 3))
        self.conv2_x3 = Conv2d_Bn(96, (3, 3))

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(3,3),strides=1,
                                                        padding='same')
        self.conv3_x1 = Conv2d_Bn(32, (3, 3))

    def call(self,inputs,training=None,**kwargs):
        x0 = self.conv0_x0(inputs,training=training)

        x1 = self.conv1_x1(inputs,training=training)
        x1 = self.conv1_x2(x1,training=training)

        x2 = self.conv2_x1(inputs,training=training)
        x2 = self.conv2_x2(x2,training=training)
        x2 = self.conv2_x3(x2,training=training)

        x3 = self.avg_pool(inputs)
        x3 = self.conv3_x1(x3,training=training)
        output = tf.keras.layers.concatenate([x0,x1,x2,x3],axis=-1)

        return output

class reduction_A(tf.keras.layers.Layer):
    def __init__(self):
        super(reduction_A,self).__init__()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,
                                                 padding='valid')
        self.conv1 = Conv2d_Bn(384,kernel_size=(3,3),strides=2,padding='valid')
        self.conv2 = Conv2d_Bn(64,kernel_size=(1,1))
        self.conv3 = Conv2d_Bn(96,kernel_size=(3,3))
        self.conv4 = Conv2d_Bn(96,kernel_size=(3,3),strides=2,padding='valid')

    def call(self,inputs, training=None, **kwargs):
        x1 = self.conv1(inputs)

        x2 = self.conv2(inputs)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)

        x3 = self.pool(inputs)

        output = tf.keras.layers.concatenate([x1,x2,x3],axis=-1)
        return output


class inception_B(tf.keras.layers.Layer):
    def __init__(self,filters=128):
        super(inception_B, self).__init__()
        self.conv1_1 = Conv2d_Bn(192,kernel_size=(1,1),strides=1)

        self.conv1_2_0 = Conv2d_Bn(filters,kernel_size=(1,1))
        self.conv1_2_1 = Conv2d_Bn(filters,kernel_size=(1,7))
        self.conv1_2_2 = Conv2d_Bn(192,kernel_size=(7,1))

        self.conv1_3_0 = Conv2d_Bn(filters,kernel_size=(1,1))
        self.conv1_3_1 = Conv2d_Bn(filters,kernel_size=(7,1))
        self.conv1_3_2 = Conv2d_Bn(filters,kernel_size=(1,7))
        self.conv1_3_3 = Conv2d_Bn(filters, kernel_size=(7, 1))
        self.conv1_3_4 = Conv2d_Bn(192, kernel_size=(1, 7))

        self.avg_pool = tf.keras.layers.AveragePooling2D((3,3),strides=1,
                                                         padding='same')
        self.avg_conv = Conv2d_Bn(192,kernel_size=(1,1))

    def call(self,inputs,training=None, **kwargs):
        x0 = self.conv1_1(inputs,training=training)

        x1 = self.avg_pool(inputs)
        x1 = self.avg_conv(x1,training=training)

        x2 = self.conv1_2_0(inputs,training=training)
        x2 = self.conv1_2_1(x2,training=training)
        x2 = self.conv1_2_2(x2,training=training)

        x3 = self.conv1_3_0(inputs,training=training)
        x3 = self.conv1_3_1(x3,training=training)
        x3 = self.conv1_3_2(x3,training=training)
        x3 = self.conv1_3_3(x3,training=training)
        x3 = self.conv1_3_4(x3,training=training)

        output = tf.keras.layers.concatenate([x0,x1,x2,x3],axis=-1)
        return output


class reduction_B(tf.keras.layers.Layer):
    def __init__(self):
        super(reduction_B, self).__init__()
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)
        self.conv1_0 = Conv2d_Bn(192,kernel_size=(1,1))
        self.conv1_1 = Conv2d_Bn(320,kernel_size=(3,3),strides=2,padding='valid')

        self.conv2_0 = Conv2d_Bn(192,kernel_size=(1,1))
        self.conv2_1 = Conv2d_Bn(192,kernel_size=(1,7))
        self.conv2_2 = Conv2d_Bn(192,kernel_size=(7,1))
        self.conv2_3 = Conv2d_Bn(192, kernel_size=(3, 3),strides=2,padding='valid')

    def call(self,inputs,training=None,**kwargs):
        x0 = self.pool(inputs)

        x1 = self.conv1_0(inputs,training=training)
        x1 = self.conv1_1(x1,training=training)

        x2 = self.conv2_0(inputs,training=training)
        x2 = self.conv2_1(x2,training=training)
        x2 = self.conv2_2(x2,training=training)
        x2 = self.conv2_3(x2,training=training)

        output = tf.keras.layers.concatenate([x0,x1,x2],axis=-1)
        return output

class inception_C(tf.keras.layers.Layer):
    def __init__(self):
        super(inception_C,self).__init__()
        self.conv1_0 = Conv2d_Bn(320,kernel_size=(1,1))

        self.conv2_0 = Conv2d_Bn(384,kernel_size=(1,1))
        self.conv2_1 = Conv2d_Bn(384,kernel_size=(1,3))
        self.conv2_2 = Conv2d_Bn(384, kernel_size=(3, 1))
        #We will add conv2_1 and conv2_2
        self.conv3_0 = Conv2d_Bn(448,kernel_size=(1,1))
        self.conv3_1 = Conv2d_Bn(384,kernel_size=(3,3))
        self.conv3_2 = Conv2d_Bn(384, kernel_size=(1, 3))
        self.conv3_3 = Conv2d_Bn(384, kernel_size=(3, 1))
        #we will add conv3_2 and conv3_3
        self.pool = tf.keras.layers.AveragePooling2D((3,3),strides=1,padding='same')
        self.conv_pool = Conv2d_Bn(192,kernel_size=(1,1))

    def call(self,inputs,training=None,**kwargs):
        x0 = self.conv1_0(inputs)

        x1 = self.conv2_0(inputs)
        x1_1 = self.conv2_1(x1)
        x1_2 = self.conv2_2(x1)
        x1_f = tf.keras.layers.concatenate([x1_1,x1_2],axis=-1)

        x2 = self.conv3_0(inputs)
        x2 = self.conv3_1(x2)
        x2_1 = self.conv3_2(x2)
        x2_2 = self.conv3_3(x2)
        x2_f = tf.keras.layers.concatenate([x2_1,x2_2],axis=-1)

        x3 = self.pool(inputs)
        x3 = self.conv_pool(x3)

        output = tf.keras.layers.concatenate([x0,x1_f,x2_f,x3],axis=-1)
        return output