import tensorflow as tf
import json
import sys
import net
import utils
import os

def main():
    if len(sys.argv) == 1:
        print(json.dumps({
            'error': 'No image passed'
        }, ensure_ascii=False).encode('utf8'))
        return

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, utils.model_dir)
        image = [utils.convert_image_to_2d_array(sys.argv[1])]
        prediction = sess.run(net.Y, feed_dict = {net.X: image})[0]

    prediction = map(lambda x: float(x), prediction)

    print(json.dumps({
        'prediction': list(prediction)
    }, ensure_ascii=False).encode('utf8'))

if __name__ == "__main__":
    main()
