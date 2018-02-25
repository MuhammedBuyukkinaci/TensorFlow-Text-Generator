import tensorflow as tf
import numpy as np
#hyperparameter'leri tanımlıyoruz.
#time-step gibi.
max_len = 20
step = 1
num_units = 256
learning_rate = 0.001
batch_size = 128
epoch = 50
temperature = 0.5

def read_data(file_name):
    '''
     open and read text file
    '''
    text = open(file_name, 'r',encoding="utf8").read()
    return text.lower()

def making_one_hot(text):
    '''
     one_hot hale getiriyoruz.
    '''
    #unique character sayısını buluyoruz.
    unique_chars = list(set(text))
    len_unique_chars = len(unique_chars)

    input_chars = []
    output_char = []
    #ilk 20 character(0:20) input, 21. character(20) output.  veya da (10500:10520) input, 10520 output.
    for i in range(0, len(text) - max_len, step):
        input_chars.append(text[i:i+max_len])
        output_char.append(text[i+max_len])
    #shape'i (len(input_chars), max_len, len_unique_chars) olan ve 0'lardan oluşan 3 boyutlu bir matrix oluşturuyoruz.
    train_data = np.zeros((len(input_chars), max_len, len_unique_chars))
    #shape'i (len(input_chars), len_unique_chars) olan ve 0'lardan oluşan 2 boyutlu bir matrix oluşturuyoruz
    target_data = np.zeros((len(input_chars), len_unique_chars))
    #one-hot formatına çeviriyoruz.
    for i , each in enumerate(input_chars):
        for j, char in enumerate(each):
            train_data[i, j, unique_chars.index(char)] = 1
        target_data[i, unique_chars.index(output_char[i])] = 1
    return train_data, target_data, unique_chars, len_unique_chars

def rnn(x, weight, bias, len_unique_chars):
    '''
     uygun bir formata getiriyoruz.
    '''
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, len_unique_chars])
    x = tf.split(x, max_len, 0)
    #graph eşitliklerini yazıyoruz
    cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    outputs, states = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
    prediction = tf.matmul(outputs[-1], weight) + bias
    return prediction

def sample(predicted):
    '''
     hangi harfin döneceğini softmax'a benzer bir şekilde probabilistic hale getiriyoruz.
    '''
    exp_predicted = np.exp(predicted/temperature)
    predicted = exp_predicted / np.sum(exp_predicted)
    probabilities = np.random.multinomial(1, predicted, 1)
    return probabilities

def run(train_data, target_data, unique_chars, len_unique_chars):
    '''
     placeholder'larımızı oluşturuyoruz.
    '''
    x = tf.placeholder("float", [None, max_len, len_unique_chars])
    y = tf.placeholder("float", [None, len_unique_chars])
    #Variable weight'lerini oluşturuyoruz.
    weight = tf.Variable(tf.random_normal([num_units, len_unique_chars]))
    bias = tf.Variable(tf.random_normal([len_unique_chars]))

    prediction = rnn(x, weight, bias, len_unique_chars)
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)
    cost = tf.reduce_mean(softmax)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    #tüm variable'ları initialize ediyoruz.
    init_op = tf.global_variables_initializer()
    #session'ımızı oluşturuyoruz
    sess = tf.Session()
    #variable'ları başlatıyoruz
    sess.run(init_op)
    #yukarıda belirlediğimiz batch size'a göre kaç tane batch olacağını hesaplıyoruz.
    num_batches = int(len(train_data)/batch_size)
    #aşagıdaki for loop kaç epoch kaçıştıracağımızı söylüyor
    for i in range(epoch):
        print("--- Epoch {0}/{1} ---".format(i+1, epoch))
        count = 0
        #aşağıdaki for loop, batch'lerin çalıştırılmasını ifade ediyor.
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
        print("Seed:", seed_chars)

        #trainingimize göre 3000 tane character tahmin ediyoruz.
        for i in range(3000):
            if i > 0:
                remove_fist_char = seed[:,1:,:]
                seed = np.append(remove_fist_char, np.reshape(probabilities, [1, 1, len_unique_chars]), axis=1)
            predicted = sess.run([prediction], feed_dict = {x:seed})
            predicted = np.asarray(predicted[0]).astype('float64')[0]
            probabilities = sample(predicted)
            predicted_chars = unique_chars[np.argmax(probabilities)]
            seed_chars += predicted_chars
        print('Result:', seed_chars)
    sess.close()
if __name__ == "__main__":
    #get data from https://s3.amazonaws.com/text-datasets/nietzsche.txt
    text = read_data('cb.txt')
    train_data, target_data, unique_chars, len_unique_chars = making_one_hot(text)
    run(train_data, target_data, unique_chars, len_unique_chars)
