import tensorflow as tf

if __name__ == '__main__':
    # print("Hello World")
    # foo = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(foo[tf.newaxis, :, :].eval())
    # x,y = [j for i in range(0,4) for j in range(0,i)], [j for i in range(0,4) for j in range(0,i)]
    print()
    D_x_layers, D_Gz_layers = [j for i in Dk_x for j in i], [j for i in Dk_Gz for j in i]
    feature_matching_loss = tf.reduce_sum([tf.reduce_mean(tf.abs(Dkx - Dkz)) for Dkx, Dkz in zip(D_x_layers, D_Gz_layers)])