from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from model.model_lstm import LSTM
import tqdm


def train():
    batch_size = 50
    num_steps = 28
    num_units = 256
    keep_prob = 0.7
    num_layers = 2
    is_training = True
    epoches = 1000

    data = input_data.read_data_sets('./data', one_hot=True)
    img_num = len(data.train.images)
    print('img num is: {}'.format(img_num))

    lstm = LSTM(batch_size, num_steps, num_units, keep_prob, num_layers, is_training)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tensorboard_writer = tf.summary.FileWriter('./tensorboard')
        tensorboard_writer.add_graph(sess.graph)
        for epoch in range(epoches):
            for step in tqdm.trange(img_num // batch_size):
                    images, labels = data.train.next_batch(batch_size)
                    _, loss, accuracy, merged_summary, global_step = sess.run(
                        [lstm.train_op, lstm.loss, lstm.accuracy, lstm.merged_summary, lstm.global_step],
                        feed_dict={
                            lstm.input_images: images,
                            lstm.input_labels: labels
                        }
                    )
                    if step %100 is 0:
                        tensorboard_writer.add_summary(merged_summary, global_step=global_step)


if __name__ == "__main__":
    train()
