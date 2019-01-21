import tensorflow as tf
import shutil
import json
import utils
import net
import os

def main():
    saver = tf.train.Saver()
    if os.path.isdir('model'):
        shutil.rmtree('model')

    #Add labels to saved model as json file for classification
    os.makedirs('model')
    with open(utils.train_data_path, 'w') as file:
        train_data = {
            'labels': utils.labels
        }
        file.write(json.dumps(train_data))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_images, test_labels = utils.random_batch(utils.config['training']['batch_size_test'])

        last_acc = 0
        for i in range(utils.config['training']['n_epoch']+1):
            batch_X, batch_Y = utils.random_batch(utils.config['training']['batch_size'])

            # the back-propagation training step
            sess.run(net.train_step, feed_dict={net.X: batch_X, net.Y_: batch_Y, net.pkeep: utils.config['training']['pkeep']})
            acc, loss = sess.run([net.accuracy, net.cross_entropy], feed_dict={net.X: test_images, net.Y_: test_labels, net.pkeep: 1.0})

            #Model accuracy has improved since last epoch
            if acc > last_acc:
                last_acc = acc

                #Remove older save
                for file_name in os.listdir(os.path.join(utils.root_dir, 'model')):
                    file_path = os.path.join(utils.root_dir, 'model', file_name)
                    if os.path.isfile(file_path) and file_name != utils.train_data_name:
                        os.unlink(file_path)

                #Save new model
                save_path = saver.save(sess, utils.model_dir)

            #Display information obout progress
            print('{0}\r'.format(' '*117), end='')
            if i%utils.config['training']['display_step'] == 0:
                print("#%d Accuracy=%.2f%%, loss=%.2f" % (i,acc*100,loss*100))
            utils.update_progress(i/utils.config['training']['n_epoch'])

if __name__ == "__main__":
    main()
