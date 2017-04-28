import tensorflow as tf
import numpy as np

#set hyperparameters
max_len = 40
step = 2
num_units = 128
learning_rate = 0.001
batch_size = 200
epoch = 60
temperature = 0.5

def read_data(file_name):
    '''
     open and read text file
    '''
    text = open(file_name, 'r').read()
    return text.lower()

def featurize(text):
    '''
     featurize the text to train and target dataset
    '''
    unique_chars = list(set(text))
    len_unique_chars = len(unique_chars)

    input_chars = []
    output_char = []

    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])

    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    target_data = np.zeros((len(input_chars), len_unique_chars))

    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

def rnn(x, weight, bias, len_unique_chars):
    '''
     define rnn cell and prediction
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_unique_chars])
    x = tf.split(x, max_len, 0)

    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction

def sample(predicted):
    '''
     helper function to sample an index from a probability array
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

def run(train_data, target_data, unique_chars, len_unique_chars):
    '''
     main run function
    '''
    x = tf.placeholder("float", [None, max_len, len_unique_chars])
    y = tf.placeholder("float", [None, len_unique_chars])
    weight = tf.Variable(tf.random_normal([num_units, len_unique_chars]))
    bias = tf.Variable(tf.random_normal([len_unique_chars]))

    prediction = rnn(x, weight, bias, len_unique_chars)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    num_batches = int(len(train_data)/batch_size)

    for i in range(epoch):
        print "----------- Epoch {0}/{1} -----------".format(i+1, epoch)
        count = 0
        for _ in range(num_batches):
            train_batch, target_batch = train_data[count:count+batch_size], target_data[count:count+batch_size]
            count += batch_size
            sess.run([optimizer] ,feed_dict={x:train_batch, y:target_batch})

        #get on of training set as seed
        seed = train_batch[:1:]

        #to print the seed 40 characters
        seed_chars = ''
        for each in seed[0]:
                seed_chars += unique_chars[np.where(each == max(each))[0][0]]
        print "Seed:", seed_chars

        #predict next 1000 characters
        for i in range(1000):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print 'Result:', seed_chars
    sess.close()


if __name__ == "__main__":
    #get data from https://s3.amazonaws.com/text-datasets/nietzsche.txt
    text = read_data('nietzsche.txt')
    train_data, target_data, unique_chars, len_unique_chars = featurize(text)
    run(train_data, target_data, unique_chars, len_unique_chars)
