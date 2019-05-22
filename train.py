from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from model.model_lstm import LSTM


# def load_data():
    

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

    for epoch in range(epoches):
        for step in range(img_num // batch_size):
                pass



if __name__ == "__main__":
    train()
    