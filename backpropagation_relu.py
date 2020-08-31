from numpy import exp, array, random, dot
import numpy as np

random.seed(1)
#synaptic_weights = 2 * random.random((3, 1)) - 1

training_set_inputs = array([[1, 0, 0], [1, 0.1, 0.2], [1, 0.31, 0.31], [1, 0, 0.1], [1, 0.81, 0], [1, 0.5, 0.1], [1, 0.1, 0], [1, 0.1, 0.1], [1, 0.5, 0.1], [1, 0.2, 0], [0, 0.5, 0.5], [0, 0.8, 1], [0, 1, 0.6], [0, 1, 1]])

hidden_weight_1 = random.random((3, 1))
hidden_weight_2 = random.random((3, 1))

output_weight = random.random((2, 1))

training_set_outputs = array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
#training_set_outputs = array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

def sigmoid_raw(v):
    return (1 / (1 + np.exp(-1 * v)))

def sigmoid(inputs, weights):
    v = dot(inputs, weights)
    return sigmoid_raw(v)

def sigmoid_d(v):
    return sigmoid_raw(v) * (1- sigmoid_raw(v)) 

def Relu(inputs, weights):
    t = dot(inputs, weights)
    r = []
    for i in range(len(t)):
        if t[i] > 0: r.append(array(t[i]))
        else: r.append(array([0]))
    return r

def Relu_d(inputs):
    r = []
    for i in range(len(inputs)):
        if inputs[i] > 0: r.append(array([1]))
        else: r.append(array([0]))
    return r

def forward(inputs):
    #forward
    o_h_1 = Relu(inputs, hidden_weight_1)
    o_h_2 = Relu(inputs, hidden_weight_2)
    '''
    o_h_1 = sigmoid(inputs, hidden_weight_1)
    o_h_2 = sigmoid(inputs, hidden_weight_2)
    '''

    o_h = array([o_h_1, o_h_2]).T[0]
    #print(o_h)
    out = Relu(o_h, output_weight)
    #out = sigmoid(o_h, output_weight)
    #out = dot(o_h, output_weight)

    return o_h_1, o_h_2, o_h, out
def loss(pred, y):
    b = []
    for i in range(len(y)):
        b.append( (pred[i] - y[i] ) )
    return b
def get_loss_val(b):
    sum = 0.0
    for i in range(len(b)):
        sum += b[i]
    return sum / len(b)
'''
learning_rate = 2e-2
iter = 50000
'''
learning_rate = 2e-4
iter = (10 ** 5)
epoch =2
for ep in range(epoch):
    print("epoch", ep +1)
    for iteration in range(iter):
        #forward
        o_h_1, o_h_2, o_h, out = forward(training_set_inputs)
        #backward(o_h_1, o_h_2, o_h, out)
        #loss' * act(out)'* out_in
        
        b = loss(out, training_set_outputs)
        #d = sigmoid_d(out)
        d = Relu_d(out)

        #d_loss = []
        sum = 0.0
        for i in range(len(b)):
            s1 = (o_h_1[i] * b[i])
            #s2 = s[i] * s1
            s2 = d[i] * s1
            sum += s2
            #d_loss.append(s2)
        
        ow = sum / len(b)
        output_weight[0] -= learning_rate * ow

        sum = 0.0
        for i in range(len(b)):
            s1 = (o_h_2[i] * b[i])
            #s2 = s[i] * s1
            s2 = d[i] * s1
            sum += s2
            #d_loss.append(s2)
        
        ow = sum / len(b)
        output_weight[1] -= learning_rate * ow
        
        pre_bp = []
        for i in range(len(b)):
            pre_bp.append(b[i] * d[i])

        #hidden_weight_1[0] update
        #p * output_weight[0] * relu(o_h_1)' * input[n]
        dr = Relu_d(o_h_1) 
        #dr = sigmoid_d(training_set_inputs, hidden_weight_1) 
        for n in range(len(hidden_weight_1)):
            sum = 0.0
            for i in range(len(training_set_inputs)):
                #loss' * relu'
                s = pre_bp[i] * output_weight[0] * dr[i] * training_set_inputs[i][n]
                sum += s
            
            ow = sum  / len(training_set_inputs)
            hidden_weight_1[n] -= learning_rate * ow


        #hidden_weight_1[1] update
        #p * output_weight[1] * relu(o_h_1)' * input[n]
        dr = Relu_d(o_h_2) 
        #dr = sigmoid_d(training_set_inputs, hidden_weight_1) 
        for n in range(len(hidden_weight_2)):
            sum = 0.0
            for i in range(len(training_set_inputs)):
                #loss' * relu'
                s = pre_bp[i] * output_weight[1] * dr[i] * training_set_inputs[i][n]
                sum += s
            
            ow = sum  / len(training_set_inputs)
            hidden_weight_2[n] -= learning_rate * ow

        if(iteration % 1000 == 0):
            print("epoch", ep+1, "iteration", iteration, "loss", get_loss_val(b))
        
    print("Out W", output_weight)
    print("H1 W", hidden_weight_1)
    print("H2 W", hidden_weight_2)

o_h_1, o_h_2, o_h, out = forward(training_set_inputs)
print(out, learning_rate)

#print (Relu(test, synaptic_weights)[0])