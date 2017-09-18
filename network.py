import numpy as np
import tensorflow as tf
import pickle

class LSTM:
    def __init__(self, hidden_size, num_cells, dict_path, text_path, lrate):
        self.graph = tf.Graph()
        self.hidden_size = hidden_size
        self.num_cells = num_cells
        self.dict_path=dict_path
        self.text_path = text_path
        self.lrate = lrate

    def createDict(self):
        with open('data/text', 'rb') as tfl:
            self.text = pickle.loads(tfl.read())
        with open('data/dictionary', 'rb') as df:
            self.dictionary = pickle.loads(df.read())
        with open('data/reverse-dictionary', 'rb') as rdf:
            self.reverse_dictionary = pickle.loads(rdf.read())
        self.vocabulary_size = len(self.dictionary)

    def createVariables(self):
        with self.graph.as_default():
            self.in_weights = {
                'forget gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.vocabulary_size],dtype=tf.float32)),
                'input gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.vocabulary_size],dtype=tf.float32)),
                'tanh gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.vocabulary_size],dtype=tf.float32)),
                'output gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.vocabulary_size],dtype=tf.float32)),
            }
            self.out_weights = {
                'forget gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.hidden_size],dtype=tf.float32)),
                'input gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.hidden_size],dtype=tf.float32)),
                'tanh gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.hidden_size],dtype=tf.float32)),
                'output gate':tf.Variable(tf.truncated_normal(shape=[self.hidden_size,self.hidden_size],dtype=tf.float32))
            }
            self.biases = {
                'forget gate':tf.Variable(tf.zeros(shape=[self.hidden_size,1]),dtype=tf.float32),
                'input gate':tf.Variable(tf.zeros(shape=[self.hidden_size,1]),dtype=tf.float32),
                'tanh gate':tf.Variable(tf.zeros(shape=[self.hidden_size,1]),dtype=tf.float32),
                'output gate':tf.Variable(tf.zeros(shape=[self.hidden_size,1]),dtype=tf.float32)
            }
            self.softmax_weights = tf.Variable(tf.truncated_normal(shape=[self.vocabulary_size,self.hidden_size],dtype=tf.float32))
            self.softmax_biases = tf.Variable(tf.zeros(shape=[self.vocabulary_size,1]),dtype=tf.float32)
            self.init_op1 = tf.global_variables_initializer()

    def defineModel1(self): #single update model
        with self.graph.as_default():
            self.y_input_ = tf.placeholder(dtype=tf.int32)
            self.y = tf.reshape(tf.one_hot(self.y_input_,self.vocabulary_size),[self.vocabulary_size,1])
            self.x_input_ = tf.placeholder(dtype=tf.int32)
            self.x_input = tf.reshape(tf.one_hot(self.x_input_,self.vocabulary_size),[self.vocabulary_size,1])
            self.cell_input = tf.placeholder(shape=[self.hidden_size,1],dtype=tf.float32)
            self.prev_input = tf.placeholder(shape=[self.hidden_size,1],dtype=tf.float32)
            forget_gate = tf.nn.sigmoid(
                    tf.matmul(self.in_weights['forget gate'],self.x_input) +
                    tf.matmul(self.out_weights['forget gate'],self.prev_input) +
            self.biases['forget gate'])
            input_gate = tf.nn.sigmoid(
                    tf.matmul(self.in_weights['input gate'], self.x_input) +
                    tf.matmul(self.out_weights['input gate'], self.prev_input) +
                self.biases['input gate']
                )
            tanh_gate = tf.nn.tanh(
                    tf.matmul(self.in_weights['tanh gate'], self.x_input) +
                    tf.matmul(self.out_weights['tanh gate'], self.prev_input) +
                self.biases['tanh gate']
            )
            self.cell_state = tf.multiply(forget_gate,self.cell_input) + tf.multiply(input_gate, tanh_gate)
            output_gate = tf.nn.sigmoid(
                tf.matmul(self.in_weights['output gate'], self.x_input) +
                tf.matmul(self.out_weights['output gate'], self.prev_input) +
            self.biases['output gate']
            )
            self.prev_cell_output = tf.multiply(output_gate, tf.nn.tanh(self.cell_state))
            self.prediction = tf.argmax(tf.nn.softmax(tf.matmul(self.softmax_weights, self.prev_cell_output)+self.softmax_biases,dim=0))
            self.pre = tf.nn.softmax(tf.matmul(self.softmax_weights, self.prev_cell_output)+self.softmax_biases,dim=0)
            self.loss = tf.abs(tf.reduce_sum(tf.subtract(self.pre,self.y,)))
            #self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=(tf.matmul(self.softmax_weights, self.prev_cell_output)+self.softmax_biases),labels=self.y)
            self.optimizer1 = tf.train.AdagradOptimizer(learning_rate=self.lrate).minimize(self.loss)
            self.initialize = tf.global_variables_initializer()
    def generateBatch(self, iteration):
        all_words = self.text[iteration:iteration+self.num_cells+1]
        return [self.dictionary[w] for w in all_words]
    def train(self, iterations):
        self.createVariables()
        self.defineModel1()
        with self.graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.8
            sess = tf.Session(config=config)
            sess.run(self.init_op1)
            sess.run(self.initialize)
            total_loss = 0
            for i in range(iterations):
                all_data = self.generateBatch(i)
                inp, correct = all_data[0], all_data[1:]
                start_cell, start_prev = np.zeros(shape=[self.hidden_size,1]), np.zeros(shape=[self.hidden_size,1])
                _, loss_val, output, state, pred = sess.run(fetches=[self.optimizer1, self.loss, self.prev_cell_output, self.cell_state, self.prediction], feed_dict={
                    self.x_input_:inp, self.y_input_:correct[0], self.cell_input:start_cell, self.prev_input:start_prev
                })
                cells_loss=loss_val
                total_loss += cells_loss
                if i%100==0 and i!=0:
                    print('network loss: ' + str(total_loss))
                    total_loss=0
                for cell in range(1,self.num_cells):
                    _, lv, output, state, pred = sess.run(
                        fetches=[self.optimizer1, self.loss, self.prev_cell_output, self.cell_state, self.prediction],
                        feed_dict={
                            self.x_input_: int(pred), self.y_input_: correct[cell], self.cell_input: state,
                            self.prev_input: output
                        })
                    cells_loss+=lv

x = LSTM(512,32,'data/dictionary','data/text',.1)
x.createDict()
x.createVariables()
x.defineModel1()
x.train(100000)
