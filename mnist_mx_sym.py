import mxnet as mx
import random
import time

from mxnet.metric import Accuracy
from mxnet.test_utils import get_mnist_iterator


BATCH_SIZE_PER_REPLICA = 512
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2
NUM_CLASSES = 10
EPOCHS = 10

mx.random.seed(42)
random.seed(42)

# pin GPUs
ctx = mx.gpu(0)

# get data
input_shape = (1, 28, 28)
train_data, test_data = get_mnist_iterator(input_shape=input_shape,
                                           batch_size=BATCH_SIZE)
# build model
data = mx.sym.Variable('data')
conv1 = mx.sym.Convolution(data=data, num_filter=32, kernel=(3,3))
relu1 = mx.sym.Activation(data=conv1, act_type="relu")
conv2 = mx.sym.Convolution(data=relu1, num_filter=64, kernel=(3,3))
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
pool = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2))
dropout1 = mx.sym.Dropout(data=pool, p=0.25)
flatten = mx.sym.Flatten(data=dropout1)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=128)
relu3 = mx.sym.Activation(data=fc1, act_type="relu")
dropout2 = mx.sym.Dropout(data=relu3, p=0.5)
fc2 = mx.sym.FullyConnected(data=dropout2, num_hidden=NUM_CLASSES)
out = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
model = mx.mod.Module(out, context=ctx)
model.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)


# initialize parameters
model.init_params(initializer=mx.init.Xavier(magnitude=2.))
opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
opt = mx.optimizer.create('adam', **opt_params)
model.init_optimizer(kvstore='device',
                     optimizer=opt)
metric = Accuracy()
# train
start = time.perf_counter()
for epoch in range(1, EPOCHS+1):
    train_data.reset()
    for i, batch in enumerate(train_data):
        if i == 0:
            tick_0 = time.time()
        model.forward(batch, is_train=True)
        model.backward()
        model.update()
        model.update_metric(metric, batch.label)
    str1 = 'Epoch [{}], Accuracy {:.4f}'.format(epoch, metric.get()[1])
    str2 = '~Samples/Sec {:.4f}'.format(BATCH_SIZE*(i+1)/(time.time()-tick_0))
    print('%s  %s' % (str1, str2))
    metric.reset()

elapsed = time.perf_counter() - start
print('elapsed: {:0.3f}'.format(elapsed))

metric = Accuracy()
model.score(test_data, metric)
print('validation %s=%f' % metric.get())
