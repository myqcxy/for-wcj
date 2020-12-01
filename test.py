import tensorflow as tf

if __name__ == '__main__':
    # print("Hello World")
    # foo = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(foo[tf.newaxis, :, :].eval())
    # x,y = [j for i in range(0,4) for j in range(0,i)], [j for i in range(0,4) for j in range(0,i)]
    # print()
    # D_x_layers, D_Gz_layers = [j for i in Dk_x for j in i], [j for i in Dk_Gz for j in i]
    # feature_matching_loss = tf.reduce_sum([tf.reduce_mean(tf.abs(Dkx - Dkz)) for Dkx, Dkz in zip(D_x_layers, D_Gz_layers)])
    # t = tf.constant([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])
    # # t = tf.constant([1,2,3,4,5,6,7,8,9])
    # # print(tf.shape(t))
    # # print(tf.size(t))
    # tf.concat([t])
    #
    # print(t)
    # # tf.reshape(t, [3, -1])
    # print(t)
    # a = tf.ones([2,3,4],tf.float32)
    a = tf.fill([2,3,4],1/8)
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    con2 = tf.concat([t1, t2], 0)

    shape2 = tf.shape(con2)
    with tf.Session() as sess:
        print(sess.run(a))
        print(sess.run(con2))
        print(sess.run(shape2))
