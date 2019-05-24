import tensorflow as tf

class LSTM(object):
    def __init__(self, batch_size, num_steps=28, num_units=256, keep_prob=0.8, num_layers=2, is_training=True):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_units = num_units
        self.keep_prob = keep_prob
        self.num_layers = num_layers
        self.is_training = is_training

        self.input_images = tf.placeholder(tf.float32, shape=[self.batch_size, 784], name='input_images')
        self.input_labels = tf.placeholder(tf.float32, shape=[self.batch_size, 10], name='input_labels')
        self.outputs, self.state = self.reference(self.input_images)
        if is_training is not True:
            return
        self.loss, self.accuracy = self.loss(self.outputs, self.input_labels)
        self.train_op = self.get_train_op(self.loss)
        self.merged_summary = self.get_summaries()

    def reference(self, x):
        with tf.variable_scope('lstm'):
            with tf.variable_scope('data_input'):
                input_images = tf.reshape(x, shape=[-1, self.num_steps])
                weights = tf.get_variable('weights', shape=[self.num_steps, self.num_units])
                biases = tf.get_variable('biases', shape=[self.num_units])
                inputs = tf.nn.xw_plus_b(input_images, weights, biases)
                inputs = tf.reshape(inputs, [-1, self.num_steps, self.num_units])

            with tf.variable_scope('cell'):
                if self.is_training is True:
                    inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)
                cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units)
                if self.is_training is True:
                    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                if self.num_layers > 1:
                    cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(self.num_layers)])
            with tf.variable_scope('dynamic_rnn'):
                initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
                # outputs shape should be [batch_size, 28, 256]
                outputs, state = tf.nn.dynamic_rnn(
                        cell=cell, inputs=inputs, initial_state=initial_state, dtype=tf.float32
                    )
        with tf.variable_scope('data_output'):
            # outputs = tf.reshape(outputs, shape=[-1, self.num_units])
            outputs = outputs[:, -1, :]
            weights = tf.get_variable('weights', shape=[self.num_units, 10])
            baises = tf.get_variable('baises', shape=[10])
            outputs = tf.nn.xw_plus_b(outputs, weights, baises)
            outputs = tf.nn.softmax(outputs)
        return outputs, state

    def loss(self, outputs, labels):
        with tf.variable_scope('loss'):
            ent = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
            ent = tf.reduce_mean(ent)
        with tf.variable_scope('accuracy'):
            logits = tf.math.argmax(outputs, axis=1)
            labels = tf.math.argmax(labels, axis=1)
            _eq = tf.equal(logits, labels)
            accuracy = tf.reduce_mean(tf.cast(_eq, tf.float32))
        return ent, accuracy

    def get_train_op(self, loss):
        tvars = tf.trainable_variables()
        self.learning_rate = tf.Variable(0.01, name='learning_rate')
        self.global_step = tf.Variable(0, name='global_step')
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        grads = tf.gradients(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, 5)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=self.global_step
        )
        return train_op

    def get_summaries(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary = tf.summary.merge_all()
        return merged_summary

