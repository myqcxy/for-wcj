""" Modular components of computational graph
    JTan 2018
"""
import tensorflow as tf
from utils import Utils
from keras.layers.convolutional import AtrousConvolution2D
from keras.layers.convolutional import Convolution2D
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

class Network(object):

    @staticmethod
    def encoder(x, config, training, C, reuse=False, actv=tf.nn.relu, scope='image'):
        """
        Process image x ([512,1024]) into a feature map of size W/16 x H/16 x C
         + C:       Bottleneck depth, controls bpp
         + Output:  Projection onto C channels, C = {2,4,8,16}
        """
        init = tf.contrib.layers.xavier_initializer()
        print('<------------ Building global {} generator architecture ------------>'.format(scope))

        def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=actv, init=init):
            bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            in_kwargs = {'center':True, 'scale': True}
            x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
            # x = tf.layers.batch_normalization(x, **bn_kwargs)
            x = tf.contrib.layers.instance_norm(x, **in_kwargs)
            x = actv(x)
            return x

        with tf.variable_scope('encoder_{}'.format(scope), reuse=reuse):

            # Run convolutions
            f = [60, 120, 240, 480, 960]
            x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            out = conv_block(x, filters=f[0], kernel_size=7, strides=1, padding='VALID', actv=actv)

            out = conv_block(out, filters=f[1], kernel_size=3, strides=2, actv=actv)
            out = conv_block(out, filters=f[2], kernel_size=3, strides=2, actv=actv)
            out = conv_block(out, filters=f[3], kernel_size=3, strides=2, actv=actv)
            out = conv_block(out, filters=f[4], kernel_size=3, strides=2, actv=actv)

            # Project channels onto space w/ dimension C
            # Feature maps have dimension W/16 x H/16 x C
            out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            feature_map = conv_block(out, filters=C, kernel_size=3, strides=1, padding='VALID', actv=actv)
            
            return feature_map


    @staticmethod
    def quantizer(w, config, reuse=False, temperature=1, L=5, scope='image'):
        """
        Quantize feature map over L centers to obtain discrete $\hat{w}$
         + Centers: {-2,-1,0,1,2}
         + TODO:    Toggle learnable centers?
        """
        with tf.variable_scope('quantizer_{}'.format(scope, reuse=reuse)):

            centers = tf.cast(tf.range(-2,3), tf.float32)
            # Partition W into the Voronoi tesellation over the centers
            w_stack = tf.stack([w for _ in range(L)], axis=-1)
            w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)

            smx = tf.nn.softmax(-1.0/temperature * tf.abs(w_stack - centers), dim=-1)
            # Contract last dimension
            w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)  # w_soft = tf.tensordot(smx, centers, axes=((-1),(0)))

            # Treat quantization as differentiable for optimization
            w_bar = tf.round(tf.stop_gradient(w_hard - w_soft) + w_soft)

            return w_bar

    @staticmethod
    def cbam_module(w,config, reuse=False, reduction_ratio=0.5, name=""):
        with tf.variable_scope("cbam_" + name, reuse=tf.AUTO_REUSE):
            hidden_num = w.get_shape().as_list()[3]
            batch_size = w.get_shape().as_list()[0]
            ###通道attension机制模块
            maxpool_channel = tf.reduce_max(tf.reduce_max(w, axis=1, keepdims=True), axis=2, keepdims=True)
            avgpool_channel = tf.reduce_mean(tf.reduce_mean(w, axis=1, keepdims=True), axis=2, keepdims=True)

            maxpool_channel = tf.layers.Flatten()(maxpool_channel)
            avgpool_channel = tf.layers.Flatten()(avgpool_channel)
            ###共享多层感知器（MLP）
            mlp_1_max = tf.layers.dense(maxpool_channel, units=int(hidden_num * reduction_ratio),
                                        name="mlp_1", reuse=None, activation=tf.nn.relu)
            mlp_2_max = tf.layers.dense(mlp_1_max, units=hidden_num, name="mlp_2", reuse=None)
            mlp_2_max = tf.reshape(mlp_2_max, [-1, 1, 1, hidden_num])  # batch_size设置为-1，就是让其自己计算具体值

            mlp_1_avg = tf.layers.dense(avgpool_channel, units=int(hidden_num * reduction_ratio),
                                        name="mlp_1", reuse=True, activation=tf.nn.relu)
            mlp_2_avg = tf.layers.dense(mlp_1_avg, units=hidden_num, name="mlp_2", reuse=True)
            mlp_2_avg = tf.reshape(mlp_2_avg, [-1, 1, 1, hidden_num])

            channel_attention = tf.nn.sigmoid(mlp_2_max + mlp_2_avg)

            channel_refined_feature = w * channel_attention

            ###空间attension机制模块
            maxpool_spatial = tf.reduce_max(channel_refined_feature, axis=3, keepdims=True)
            avgpool_spatial = tf.reduce_mean(channel_refined_feature, axis=3, keepdims=True)
            max_avg_pool_spatial = tf.concat([maxpool_spatial, avgpool_spatial], axis=3)
            conv_layer = tf.layers.conv2d(max_avg_pool_spatial, filters=1, kernel_size=(7, 7),
                                          padding="same", activation=None)
            spatial_attention = tf.nn.sigmoid(conv_layer)

            refined_feature = channel_refined_feature * spatial_attention
            # print('===================wwwwwwwwwwwwwwwwwwwwwwwwwww==========================')
            # print(type(refined_feature))
            # print(refined_feature.dtype)
            # print(refined_feature.shape)
            # print(refined_feature)

        return refined_feature

    @staticmethod
    def rbac_module(refined_feature, reuse=False, actv=tf.nn.relu, scope='image'):
        init = tf.contrib.layers.xavier_initializer()

        def residual_block(x, n_filters, kernel_size=3, strides=1, actv=actv):
            init = tf.contrib.layers.xavier_initializer()
            # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            strides = [1, 1]
            identity_map = x

            p = int((kernel_size - 1) / 2)
            res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                                   activation=None, padding='VALID')
            res = actv(tf.contrib.layers.instance_norm(res))

            res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                                   activation=None, padding='VALID')
            res = tf.contrib.layers.instance_norm(res)

            assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
            out = tf.add(res, identity_map)

            return out

        with tf.variable_scope('rbac_module_{}'.format(scope, reuse=reuse)):
            refined_feature = tf.pad(refined_feature, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            refined_feature = tf.layers.conv2d(refined_feature, filters=128, kernel_size=7, strides=1, padding='VALID')
            # refined_feature = tf.layers.conv2d(refined_feature, filters=128, kernel_size=3*3, strides=1,
            #         activation=tf.nn.relu, padding='SAME')

            # refined_feature = tf.layers.conv2d(refined_feature, filters=256, kernel_size=3*3, strides=1,  #加一个卷积层，使得进入下一个模块输入输出tensor一致
            #         activation=tf.nn.relu, padding='SAME')
            # refined_feature = AtrousConvolution2D(128, 3, atrous_rate = (3,3),padding = 'SAME',name = 'atrous_conv1' )(refined_feature)
            # refined_feature = AtrousConvolution2D(256, 3, atrous_rate = (3,3),padding = 'SAME',name = 'atrous_conv2' )(refined_feature)
            refined_feature = Convolution2D(128, 3, dilation_rate=(3, 3), padding='SAME', name='atrous_conv1')(
                refined_feature)
            refined_feature = Convolution2D(256, 3, dilation_rate=(3, 3), padding='SAME', name='atrous_conv2')(
                refined_feature)
            res = residual_block(refined_feature, 256, actv=actv)
            res = residual_block(res, 256, actv=actv)

            # y = AtrousConvolution2D(1, 3, atrous_rate = (3,3),padding = 'SAME',name = 'atrous_conv3' )(res)
            y = Convolution2D(1, 3, dilation_rate=(3, 3), padding='SAME', name='atrous_conv3')(res)
            # y = tf.layers.conv2d(res, filters=1, kernel_size=3*3, strides=1, activation=None, padding='SAME')
            print('===================wwwwwwwwwwwwwwwwwwwwwwwwwww==========================')
            print(type(y))
            print(y.dtype)
            print(y.shape)
            return y
    # def rbac_module(refined_feature,config, reuse=False, actv=tf.nn.relu, scope='image'):
    #     init = tf.contrib.layers.xavier_initializer()
    #
    #     def residual_block(x, n_filters, kernel_size=3, strides=1, actv=actv):
    #         init = tf.contrib.layers.xavier_initializer()
    #         # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
    #         strides = [1, 1]
    #         identity_map = x
    #
    #         p = int((kernel_size - 1) / 2)
    #         res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    #         res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
    #                                activation=None, padding='VALID')
    #         res = actv(tf.contrib.layers.instance_norm(res))
    #
    #         res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
    #         res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
    #                                activation=None, padding='VALID')
    #         res = tf.contrib.layers.instance_norm(res)
    #
    #         assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
    #         out = tf.add(res, identity_map)
    #
    #         return out
    #
    #     with tf.variable_scope('rbac_module_{}'.format(scope, reuse=reuse)):
    #         refined_feature = tf.pad(refined_feature, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
    #         # refined_feature = tf.layers.conv2d(refined_feature, filters=256, kernel_size=3*3, strides=1,  #加一个卷积层，使得进入下一个模块输入输出tensor一致
    #         #        activation=tf.nn.relu, padding='SAME')
    #         # refined_feature = AtrousConvolution2D(128, 3, atrous_rate = (3,3),padding = 'SAME',name = 'atrous_conv1' )(refined_feature)
    #         # refined_feature = AtrousConvolution2D(256, 3, atrous_rate = (3,3),padding = 'SAME',name = 'atrous_conv2' )(refined_feature)
    #         refined_feature = Convolution2D(128, 3, dilation_rate=(3, 3), padding='SAME', name='atrous_conv1')(
    #             refined_feature)
    #         refined_feature = Convolution2D(256, 3, dilation_rate=(3, 3), padding='SAME', name='atrous_conv2')(
    #             refined_feature)
    #         res = residual_block(refined_feature, 256, actv=actv)
    #         res = residual_block(res, 256, actv=actv)
    #
    #         # y = AtrousConvolution2D(1, 3, atrous_rate = (3,3),padding = 'SAME',name = 'atrous_conv3' )(res)
    #         y = Convolution2D(1, 3, dilation_rate=(3, 3), padding='SAME', name='atrous_conv3')(res)
    #         # y = tf.layers.conv2d(res, filters=1, kernel_size=3*3, strides=1,activation=None, padding='VALID')
    #         print('===================wwwwwwwwwwwwwwwwwwwwwwwwwww==========================')
    #         print(type(y))
    #         print(y.dtype)
    #         print(y.shape)
    #         return y

    @staticmethod
    def masker_module_use_tf(y, config, reuse=False, C=8, scope='image'):
        with tf.variable_scope('masker_module_{}'.format(scope, reuse=reuse)):
            def test(a,batch_size,height,width,channels,value):
                a = tf.cast(a, tf.float32)
                b = tf.fill([batch_size, height, width, channels], value)

                b = tf.cast(b, tf.float32)
                # zeros = tf.zeros([batch_size,height,width,channels],dtype=tf.float32)
                # c = tf.subtract(a,b)
                c = tf.greater_equal(a,b)
                return tf.cast(c,tf.int32)

            (mean,variance) = tf.nn.moments(x=y,axes=[1,2])
            y_hat = tf.nn.batch_normalization(y,mean=mean,variance=variance,variance_epsilon=0.001, scale=1, offset=0) # 标准化

            m = tf.nn.sigmoid(y_hat)

            (batch_size, height, width, channels) = (1,8,16,1)
            # t = np.zeros((batch_size,height,width,8))
            s1 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=0)
            s2 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=1/8)
            s3 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=2/8)
            s4 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=3/8)
            s5 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=4/8)
            s6 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=5/8)
            s7 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=6/8)
            s8 = test(a=m,batch_size=batch_size,height=height,width=width,channels=channels,value=7/8)
            m_hat = tf.concat([s1,s2,s3,s4,s5,s6,s7,s8],3)

            return tf.cast(m_hat,tf.float32)

    @staticmethod
    def masker_module_version_2(y, config, reuse=False, C=8, scope='image'):
        with tf.variable_scope('masker_module_{}'.format(scope, reuse=reuse)):
            t = y
            init = tf.initialize_local_variables()
            session = tf.Session()
            session.run(init)
            # t = type(y.eval())
            t = session.run(t)  # 将四维tensor转化为四维numpy数组
            t = np.array(t)  # 转为矩阵表示形式
            print(y)
            t = t.reshape(-1, 8)  # 将四维矩阵转化为二维矩阵

            print(y)

            scaler = StandardScaler()  # z-score标准化（直接调用内部处理函数）Z = (X - µ) / σ
            y_hat = scaler.fit_transform(t)

            print(y_hat)
            min_max_scaler = preprocessing.MinMaxScaler()  # 最值归一化（直接调用内部处理函数）Z = [Xi - min(X)]/[max(X) - min(X)]
            m = min_max_scaler.fit_transform(y_hat)
            print(m)

            ###  将2-D内容感知重要性图扩展为3-D的  ###  扩展
            m = np.array(m)
            print(m.shape)
            (rows, cols) = m.shape
            m_hat = [[[0] * cols for _ in range(rows)] for _ in range(C)]
            m_hat = np.array(m_hat)
            print(m_hat)
            (channel, rows, cols) = m_hat.shape
            for i in range(rows):
                for j in range(cols):
                    for c in range(channel):
                        if m[i][j] < (c + 2) / C:
                            m_hat[c][i][j] = 0
                        else:
                            m_hat[c][i][j] = 1
            (channel, rows, cols) = m_hat.shape
            print(m_hat)
            print(m_hat.shape)
            m_hat = m_hat.reshape(-1, rows, cols, channel)  # 将其扩展成四维矩阵
            m_hat = tf.convert_to_tensor(m_hat)  # 转为四维tensor
            print('===================wwwwwwwwwwwwwwwwwwwwwwwwwww==========================')
            print(type(m_hat))
            print(m_hat)
            print(m_hat.get_shape())
            return (m_hat)
    @staticmethod
    def masker_module(y, config,reuse=False, C=8, scope='image'):
        with tf.variable_scope('masker_module_{}'.format(scope, reuse=reuse)):
            init = tf.initialize_all_variables()

            session = tf.Session()
            session.run(init)
            # t = type(y.eval())
            y = session.run(y)  # 将四维tensor转化为四维numpy数组
            y = np.array(y)  # 转为矩阵表示形式
            print(y)
            y = y.reshape(-1, 8)  # 将四维矩阵转化为二维矩阵

            print(y)

            scaler = StandardScaler()  # z-score标准化（直接调用内部处理函数）Z = (X - µ) / σ
            y_hat = scaler.fit_transform(y)

            print(y_hat)
            min_max_scaler = preprocessing.MinMaxScaler()  # 最值归一化（直接调用内部处理函数）Z = [Xi - min(X)]/[max(X) - min(X)]
            m = min_max_scaler.fit_transform(y_hat)
            print(m)

            ###  将2-D内容感知重要性图扩展为3-D的  ###  扩展
            m = np.array(m)
            print(m.shape)
            (rows, cols) = m.shape
            m_hat = [[[0] * cols for _ in range(rows)] for _ in range(C)]
            m_hat = np.array(m_hat)
            print(m_hat)
            (channel, rows, cols) = m_hat.shape
            for i in range(rows):
                for j in range(cols):
                    for c in range(channel):
                        if m[i][j] < (c + 2) / C:
                            m_hat[c][i][j] = 0
                        else:
                            m_hat[c][i][j] = 1
            (channel, rows, cols) = m_hat.shape
            print(m_hat)
            print(m_hat.shape)
            m_hat = m_hat.reshape(-1, rows, cols, channel)  # 将其扩展成四维矩阵
            m_hat = tf.convert_to_tensor(m_hat)  # 转为四维tensor
            print('===================wwwwwwwwwwwwwwwwwwwwwwwwwww==========================')
            print(type(m_hat))
            print(m_hat)
            print(m_hat.get_shape())
            return (m_hat)
    @staticmethod
    def decoder(w_bar, config, training, C, reuse=False, actv=tf.nn.relu, channel_upsample=960):
        """
        Attempt to reconstruct the image from the quantized representation w_bar.
        Generated image should be consistent with the true image distribution while
        recovering the specific encoded image
        + C:        Bottleneck depth, controls bpp - last dimension of encoder output
        + TODO:     Concatenate quantized w_bar with noise sampled from prior
        """
        init = tf.contrib.layers.xavier_initializer()

        def residual_block(x, n_filters, kernel_size=3, strides=1, actv=actv):
            init = tf.contrib.layers.xavier_initializer()
            # kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            strides = [1,1]
            identity_map = x

            p = int((kernel_size-1)/2)
            res = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                    activation=None, padding='VALID')
            res = actv(tf.contrib.layers.instance_norm(res))

            res = tf.pad(res, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
            res = tf.layers.conv2d(res, filters=n_filters, kernel_size=kernel_size, strides=strides,
                    activation=None, padding='VALID')
            res = tf.contrib.layers.instance_norm(res)

            assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between input/output!'
            out = tf.add(res, identity_map)

            return out

        def upsample_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=actv, batch_norm=False):
            bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
            in_kwargs = {'center':True, 'scale': True}
            x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
            if batch_norm is True:
                x = tf.layers.batch_normalization(x, **bn_kwargs)
            else:
                x = tf.contrib.layers.instance_norm(x, **in_kwargs)
            x = actv(x)

            return x

        # Project channel dimension of w_bar to higher dimension
        # W_pc = tf.get_variable('W_pc_{}'.format(C), shape=[C, channel_upsample], initializer=init)
        # upsampled = tf.einsum('ijkl,lm->ijkm', w_bar, W_pc)
        with tf.variable_scope('decoder', reuse=reuse):
            w_bar = tf.pad(w_bar, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            upsampled = Utils.conv_block(w_bar, filters=960, kernel_size=3, strides=1, padding='VALID', actv=actv)
            
            # Process upsampled feature map with residual blocks
            res = residual_block(upsampled, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)
            res = residual_block(res, 960, actv=actv)

            # Upsample to original dimensions - mirror decoder
            f = [480, 240, 120, 60]

            ups = upsample_block(res, f[0], 3, strides=[2,2], padding='same')
            ups = upsample_block(ups, f[1], 3, strides=[2,2], padding='same')
            ups = upsample_block(ups, f[2], 3, strides=[2,2], padding='same')
            ups = upsample_block(ups, f[3], 3, strides=[2,2], padding='same')
            
            ups = tf.pad(ups, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            ups = tf.layers.conv2d(ups, 3, kernel_size=7, strides=1, padding='VALID')

            out = tf.nn.tanh(ups)

            return out


    @staticmethod
    def discriminator(x, config, training, reuse=False, actv=tf.nn.leaky_relu, use_sigmoid=False, ksize=4):
        # x is either generator output G(z) or drawn from the real data distribution
        # Patch-GAN discriminator based on arXiv 1711.11585
        # bn_kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        in_kwargs = {'center':True, 'scale':True, 'activation_fn':actv}

        print('Shape of x:', x.get_shape().as_list())

        with tf.variable_scope('discriminator', reuse=reuse):
            c1 = tf.layers.conv2d(x, 64, kernel_size=ksize, strides=2, padding='same', activation=actv)
            c2 = tf.layers.conv2d(c1, 128, kernel_size=ksize, strides=2, padding='same')
            c2 = actv(tf.contrib.layers.instance_norm(c2, **in_kwargs))
            c3 = tf.layers.conv2d(c2, 256, kernel_size=ksize, strides=2, padding='same')
            c3 = actv(tf.contrib.layers.instance_norm(c3, **in_kwargs))
            c4 = tf.layers.conv2d(c3, 512, kernel_size=ksize, strides=2, padding='same')
            c4 = actv(tf.contrib.layers.instance_norm(c4, **in_kwargs))

            out = tf.layers.conv2d(c4, 1, kernel_size=ksize, strides=1, padding='same')

            if use_sigmoid is True:  # Otherwise use LS-GAN
                out = tf.nn.sigmoid(out)

        return out


    @staticmethod
    def multiscale_discriminator(x, config, training, actv=tf.nn.leaky_relu, use_sigmoid=False, 
        ksize=4, mode='real', reuse=False):
        # x is either generator output G(z) or drawn from the real data distribution
        # Multiscale + Patch-GAN discriminator architecture based on arXiv 1711.11585
        print('<------------ Building multiscale discriminator architecture ------------>')

        if mode == 'real':
            print('Building discriminator D(x)')
        elif mode == 'reconstructed':
            print('Building discriminator D(G(z))')
        else:
            raise NotImplementedError('Invalid discriminator mode specified.')

        # Downsample input
        x2 = tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='same')
        x4 = tf.layers.average_pooling2d(x2, pool_size=3, strides=2, padding='same')

        print('Shape of x:', x.get_shape().as_list())
        print('Shape of x downsampled by factor 2:', x2.get_shape().as_list())
        print('Shape of x downsampled by factor 4:', x4.get_shape().as_list())

        def discriminator(x, scope, actv=actv, use_sigmoid=use_sigmoid, ksize=ksize, reuse=reuse):

            # Returns patch-GAN output + intermediate layers

            with tf.variable_scope('discriminator_{}'.format(scope), reuse=reuse):
                c1 = tf.layers.conv2d(x, 64, kernel_size=ksize, strides=2, padding='same', activation=actv)
                c2 = Utils.conv_block(c1, filters=128, kernel_size=ksize, strides=2, padding='same', actv=actv)
                c3 = Utils.conv_block(c2, filters=256, kernel_size=ksize, strides=2, padding='same', actv=actv)
                c4 = Utils.conv_block(c3, filters=512, kernel_size=ksize, strides=2, padding='same', actv=actv)
                out = tf.layers.conv2d(c4, 1, kernel_size=ksize, strides=1, padding='same')

                if use_sigmoid is True:  # Otherwise use LS-GAN
                    out = tf.nn.sigmoid(out)

            return out, c1, c2, c3, c4

        with tf.variable_scope('discriminator', reuse=reuse):
            disc, *Dk = discriminator(x, 'original')
            disc_downsampled_2, *Dk_2 = discriminator(x2, 'downsampled_2')
            disc_downsampled_4, *Dk_4 = discriminator(x4, 'downsampled_4')

        return disc, disc_downsampled_2, disc_downsampled_4, Dk, Dk_2, Dk_4

    @staticmethod
    def dcgan_generator(z, config, training, C, reuse=False, actv=tf.nn.relu, kernel_size=5, upsample_dim=256):
        """
        Upsample noise to concatenate with quantized representation w_bar.
        + z:    Drawn from latent distribution - [batch_size, noise_dim]
        + C:    Bottleneck depth, controls bpp - last dimension of encoder output
        """
        init =  tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        with tf.variable_scope('noise_generator', reuse=reuse):

            # [batch_size, 4, 8, dim]
            with tf.variable_scope('fc1', reuse=reuse):
                h2 = tf.layers.dense(z, units=4 * 8 * upsample_dim, activation=actv, kernel_initializer=init)  # cifar-10
                h2 = tf.layers.batch_normalization(h2, **kwargs)
                h2 = tf.reshape(h2, shape=[-1, 4, 8, upsample_dim])

            # [batch_size, 8, 16, dim/2]
            with tf.variable_scope('upsample1', reuse=reuse):
                up1 = tf.layers.conv2d_transpose(h2, upsample_dim//2, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
                up1 = tf.layers.batch_normalization(up1, **kwargs)

            # [batch_size, 16, 32, dim/4]
            with tf.variable_scope('upsample2', reuse=reuse):
                up2 = tf.layers.conv2d_transpose(up1, upsample_dim//4, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
                up2 = tf.layers.batch_normalization(up2, **kwargs)
            
            # [batch_size, 32, 64, dim/8]
            with tf.variable_scope('upsample3', reuse=reuse):
                up3 = tf.layers.conv2d_transpose(up2, upsample_dim//8, kernel_size=kernel_size, strides=2, padding='same', activation=actv)  # cifar-10
                up3 = tf.layers.batch_normalization(up3, **kwargs)

            with tf.variable_scope('conv_out', reuse=reuse):
                out = tf.pad(up3, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
                out = tf.layers.conv2d(out, C, kernel_size=7, strides=1, padding='VALID')

        return out

    @staticmethod
    def dcgan_discriminator(x, config, training, reuse=False, actv=tf.nn.relu):
        # x is either generator output G(z) or drawn from the real data distribution
        init =  tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        print('Shape of x:', x.get_shape().as_list())
        x = tf.reshape(x, shape=[-1, 32, 32, 3]) 
        # x = tf.reshape(x, shape=[-1, 28, 28, 1]) 

        with tf.variable_scope('discriminator', reuse=reuse):
            with tf.variable_scope('conv1', reuse=reuse):
                c1 = tf.layers.conv2d(x, 64, kernel_size=5, strides=2, padding='same', activation=actv)
                c1 = tf.layers.batch_normalization(c1, **kwargs)

            with tf.variable_scope('conv2', reuse=reuse):
                c2 = tf.layers.conv2d(c1, 128, kernel_size=5, strides=2, padding='same', activation=actv)
                c2 = tf.layers.batch_normalization(c2, **kwargs)

            with tf.variable_scope('fc1', reuse=reuse):
                fc1 = tf.contrib.layers.flatten(c2)
                # fc1 = tf.reshape(c2, shape=[-1, 8 * 8 * 128])
                fc1 = tf.layers.dense(fc1, units=1024, activation=actv, kernel_initializer=init)
                fc1 = tf.layers.batch_normalization(fc1, **kwargs)
            
            with tf.variable_scope('out', reuse=reuse):
                out = tf.layers.dense(fc1, units=2, activation=None, kernel_initializer=init)

        return out
        

    @staticmethod
    def critic_grande(x, config, training, reuse=False, actv=tf.nn.relu, kernel_size=5, gradient_penalty=True):
        # x is either generator output G(z) or drawn from the real data distribution
        init =  tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':False}
        print('Shape of x:', x.get_shape().as_list())
        x = tf.reshape(x, shape=[-1, 32, 32, 3]) 
        # x = tf.reshape(x, shape=[-1, 28, 28, 1]) 

        with tf.variable_scope('critic', reuse=reuse):
            with tf.variable_scope('conv1', reuse=reuse):
                c1 = tf.layers.conv2d(x, 64, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
                if gradient_penalty is False:
                    c1 = tf.layers.batch_normalization(c1, **kwargs)

            with tf.variable_scope('conv2', reuse=reuse):
                c2 = tf.layers.conv2d(c1, 128, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
                if gradient_penalty is False:
                    c2 = tf.layers.batch_normalization(c2, **kwargs)

            with tf.variable_scope('conv3', reuse=reuse):
                c3 = tf.layers.conv2d(c2, 256, kernel_size=kernel_size, strides=2, padding='same', activation=actv)
                if gradient_penalty is False:
                    c3 = tf.layers.batch_normalization(c3, **kwargs)

            with tf.variable_scope('fc1', reuse=reuse):
                fc1 = tf.contrib.layers.flatten(c3)
                # fc1 = tf.reshape(c2, shape=[-1, 8 * 8 * 128])
                fc1 = tf.layers.dense(fc1, units=1024, activation=actv, kernel_initializer=init)
                #fc1 = tf.layers.batch_normalization(fc1, **kwargs)
            
            with tf.variable_scope('out', reuse=reuse):
                out = tf.layers.dense(fc1, units=1, activation=None, kernel_initializer=init)

        return out

    @staticmethod
    def wrn(x, config, training, reuse=False, actv=tf.nn.relu):
        # Implements W-28-10 wide residual network
        # See Arxiv 1605.07146
        network_width = 10 # k
        block_multiplicity = 2 # n

        filters = [16, 16, 32, 64]
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}

        def residual_block(x, n_filters, actv, keep_prob, training, project_shortcut=False, first_block=False):
            init = tf.contrib.layers.xavier_initializer()
            kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}

            if project_shortcut:
                strides = [2,2] if not first_block else [1,1]
                identity_map = tf.layers.conv2d(x, filters=n_filters, kernel_size=[1,1],
                                   strides=strides, kernel_initializer=init, padding='same')
                # identity_map = tf.layers.batch_normalization(identity_map, **kwargs)
            else:
                strides = [1,1]
                identity_map = x

            bn = tf.layers.batch_normalization(x, **kwargs)
            conv = tf.layers.conv2d(bn, filters=n_filters, kernel_size=[3,3], activation=actv,
                       strides=strides, kernel_initializer=init, padding='same')

            bn = tf.layers.batch_normalization(conv, **kwargs)
            do = tf.layers.dropout(bn, rate=1-keep_prob, training=training)

            conv = tf.layers.conv2d(do, filters=n_filters, kernel_size=[3,3], activation=actv,
                       kernel_initializer=init, padding='same')
            out = tf.add(conv, identity_map)

            return out

        def residual_block_2(x, n_filters, actv, keep_prob, training, project_shortcut=False, first_block=False):
            init = tf.contrib.layers.xavier_initializer()
            kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}
            prev_filters = x.get_shape().as_list()[-1]
            if project_shortcut:
                strides = [2,2] if not first_block else [1,1]
                # identity_map = tf.layers.conv2d(x, filters=n_filters, kernel_size=[1,1],
                #                   strides=strides, kernel_initializer=init, padding='same')
                identity_map = tf.layers.average_pooling2d(x, strides, strides, 'valid')
                identity_map = tf.pad(identity_map, 
                    tf.constant([[0,0],[0,0],[0,0],[(n_filters-prev_filters)//2, (n_filters-prev_filters)//2]]))
                # identity_map = tf.layers.batch_normalization(identity_map, **kwargs)
            else:
                strides = [1,1]
                identity_map = x

            x = tf.layers.batch_normalization(x, **kwargs)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d(x, filters=n_filters, kernel_size=[3,3], strides=strides,
                    kernel_initializer=init, padding='same')

            x = tf.layers.batch_normalization(x, **kwargs)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=1-keep_prob, training=training)

            x = tf.layers.conv2d(x, filters=n_filters, kernel_size=[3,3],
                       kernel_initializer=init, padding='same')
            out = tf.add(x, identity_map)

            return out

        with tf.variable_scope('wrn_conv', reuse=reuse):
            # Initial convolution --------------------------------------------->
            with tf.variable_scope('conv0', reuse=reuse):
                conv = tf.layers.conv2d(x, filters[0], kernel_size=[3,3], activation=actv,
                                        kernel_initializer=init, padding='same')
            # Residual group 1 ------------------------------------------------>
            rb = conv
            f1 = filters[1]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group1/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f1, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training, first_block=True)
            # Residual group 2 ------------------------------------------------>
            f2 = filters[2]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group2/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f2, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training)
            # Residual group 3 ------------------------------------------------>
            f3 = filters[3]*network_width
            for n in range(block_multiplicity):
                with tf.variable_scope('group3/{}'.format(n), reuse=reuse):
                    project_shortcut = True if n==0 else False
                    rb = residual_block(rb, f3, actv, project_shortcut=project_shortcut,
                            keep_prob=config.conv_keep_prob, training=training)
            # Avg pooling + output -------------------------------------------->
            with tf.variable_scope('output', reuse=reuse):
                bn = tf.nn.relu(tf.layers.batch_normalization(rb, **kwargs))
                avp = tf.layers.average_pooling2d(bn, pool_size=[8,8], strides=[1,1], padding='valid')
                flatten = tf.contrib.layers.flatten(avp)
                out = tf.layers.dense(flatten, units=config.n_classes, kernel_initializer=init)

            return out


    @staticmethod
    def old_encoder(x, config, training, C, reuse=False, actv=tf.nn.relu):
        """
        Process image x ([512,1024]) into a feature map of size W/16 x H/16 x C
         + C:       Bottleneck depth, controls bpp
         + Output:  Projection onto C channels, C = {2,4,8,16}
        """
        # proj_channels = [2,4,8,16]
        init = tf.contrib.layers.xavier_initializer()

        def conv_block(x, filters, kernel_size=[3,3], strides=2, padding='same', actv=actv, init=init):
            in_kwargs = {'center':True, 'scale': True}
            x = tf.layers.conv2d(x, filters, kernel_size, strides=strides, padding=padding, activation=None)
            x = tf.contrib.layers.instance_norm(x, **in_kwargs)
            x = actv(x)
            return x
                
        with tf.variable_scope('encoder', reuse=reuse):

            # Run convolutions
            out = conv_block(x, kernel_size=3, strides=1, filters=160, actv=actv)
            out = conv_block(out, kernel_size=[3,3], strides=2, filters=320, actv=actv)
            out = conv_block(out, kernel_size=[3,3], strides=2, filters=480, actv=actv)
            out = conv_block(out, kernel_size=[3,3], strides=2, filters=640, actv=actv)
            out = conv_block(out, kernel_size=[3,3], strides=2, filters=800, actv=actv)

            out = conv_block(out, kernel_size=3, strides=1, filters=960, actv=actv)
            # Project channels onto lower-dimensional embedding space
            W = tf.get_variable('W_channel_{}'.format(C), shape=[960,C], initializer=init)
            feature_map = tf.einsum('ijkl,lm->ijkm', out, W)  # feature_map = tf.tensordot(out, W, axes=((3),(0)))
            
            # Feature maps have dimension W/16 x H/16 x C
            return feature_map


