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

    print('Loading test batch...')
    test_images, test_labels = utils.random_batch(utils.config['training']['batch_size_test'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

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
            utils.display_progress(i, acc, loss)

if __name__ == "__main__":
    main()
