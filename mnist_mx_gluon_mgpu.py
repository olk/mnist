import mxnet as mx
import numpy as np
import random
import time

from mxnet import autograd as ag
from mxnet.io import NDArrayIter
from mxnet.metric import Accuracy
from mxnet.optimizer import Adam
from mxnet.test_utils import get_mnist_iterator
from mxnet.gluon import Block, Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import Conv2D, Dense, Dropout, Flatten, MaxPool2D, HybridBlock
from mxnet.gluon.utils import split_and_load


BATCH_SIZE_PER_REPLICA = 512
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2
NUM_CLASSES = 10
EPOCHS = 10
GPU_COUNT = 2


class Model(HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = Conv2D(32, (3, 3))
            self.conv2 = Conv2D(64, (3, 3))
            self.pool = MaxPool2D(pool_size=(2, 2))
            self.dropout1 = Dropout(0.25)
            self.flatten = Flatten()
            self.dense1 = Dense(128)
            self.dropout2 = Dropout(0.5)
            self.dense2 = Dense(NUM_CLASSES)

    def hybrid_forward(self, F, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.dense2(x)
        return x
   

mx.random.seed(42)
random.seed(42)

# get data
input_shape = (1, 28, 28)
train_data, test_data = get_mnist_iterator(input_shape=input_shape,
                                           batch_size=BATCH_SIZE)

# build nodel
model = Model()
# hybridize for speed
model.hybridize(static_alloc=True, static_shape=True)

# pin GPUs
ctx = [mx.gpu(i) for i in range(GPU_COUNT)]

# optimizer
opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
opt = mx.optimizer.create('adam', **opt_params)
# initialize parameters
model.initialize(force_reinit=True, ctx=ctx)
# fetch and broadcast parameters
params = model.collect_params()
# trainer
trainer = Trainer(params=params,
                  optimizer=opt,
                  kvstore='device')
# loss function
loss_fn = SoftmaxCrossEntropyLoss()
# use accuracy as the evaluation metric
metric = Accuracy()

start = time.perf_counter()
for epoch in range(1, EPOCHS+1):
    # reset the train data iterator.
    train_data.reset()
    # loop over the train data iterator
    for i, batch in enumerate(train_data):
        if i == 0:
            tick_0 = time.time()
        # splits train data into multiple slices along batch_axis
        # copy each slice into a context
        data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # splits train labels into multiple slices along batch_axis
        # copy each slice into a context
        label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        losses = []
        # inside training scope
        with ag.record():
            for x, y in zip(data, label):
                z = model(x)
                # computes softmax cross entropy loss
                l = loss_fn(z, y)
                outputs.append(z)
                losses.append(l)
        # backpropagate the error for one iteration
        for l in losses:
            l.backward()
        # make one step of parameter update.
        # trainer needs to know the batch size of data
        # to normalize the gradient by 1/batch_size
        trainer.step(BATCH_SIZE)
        # updates internal evaluation
        metric.update(label, outputs)
    str1 = 'Epoch [{}], Accuracy {:.4f}'.format(epoch, metric.get()[1])
    str2 = '~Samples/Sec {:.4f}'.format(BATCH_SIZE*(i+1)/(time.time()-tick_0))
    print('%s  %s' % (str1, str2))
    # reset evaluation result to initial state.
    metric.reset()

elapsed = time.perf_counter() - start
print('elapsed: {:0.3f}'.format(elapsed))

# use Accuracy as the evaluation metric
metric = Accuracy()
for batch in test_data:
    data = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
    label = split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
    outputs = []
    for x in data:
        outputs.append(model(x))
    metric.update(label, outputs)
print('validation %s=%f' % metric.get())
