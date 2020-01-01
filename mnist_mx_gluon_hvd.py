import horovod.mxnet as hvd
import mxnet as mx
import numpy as np
import os
import random
import time
import zipfile

from mxnet import autograd as ag
from mxnet.metric import Accuracy
from mxnet.test_utils import get_mnist_iterator
from mxnet.gluon import Block
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import Conv2D, Dense, Dropout, Flatten, MaxPool2D, HybridBlock
from mxnet.gluon.utils import split_and_load


BATCH_SIZE_PER_REPLICA = 512
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * 2
NUM_CLASSES = 10
EPOCHS = 10


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

# initialize Horovod
hvd.init()

# get data
input_shape = (1, 28, 28)
train_data, test_data = get_mnist_iterator(input_shape=input_shape,
                                           batch_size=BATCH_SIZE,
                                           num_parts=hvd.size(),
                                           part_index=hvd.rank())
# build model
model = Model()
# hybridize for speed
model.hybridize(static_alloc=True, static_shape=True)

# pin GPU to be used to process local rank
ctx = mx.gpu(hvd.local_rank())

# optimizer
opt_params={'learning_rate':0.001, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-08}
opt = mx.optimizer.create('adam', **opt_params)
# initialize parameters
model.initialize(force_reinit=True, ctx=ctx)
# fetch and broadcast parameters
params = model.collect_params()
if params is not None:
    hvd.broadcast_parameters(params, root_rank=0)
# create DistributedTrainer, a subclass of gluon.Trainer
trainer = hvd.DistributedTrainer(params, opt)
# loss function
loss_fn = SoftmaxCrossEntropyLoss()
# use accuracy as the evaluation metric
metric = Accuracy()
# train
start = time.perf_counter()
for epoch in range(1, EPOCHS+1):
    # Reset the train data iterator.
    train_data.reset()
    for i, batch in enumerate(train_data):
        if i == 0:
            tick_0 = time.time()
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        with ag.record():
            output = model(data.astype('float32', copy=False))
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(BATCH_SIZE)
        metric.update([label], [output])
    str1 = 'Epoch [{}], Accuracy {:.4f}'.format(epoch, metric.get()[1])
    str2 = '~Samples/Sec {:.4f}'.format(BATCH_SIZE*(i+1)/(time.time()-tick_0))
    print('%s  %s' % (str1, str2))
    # Reset evaluation result to initial state.
    metric.reset()

if 0 == hvd.rank():
    elapsed = time.perf_counter() - start
    print('elapsed: {:0.3f}'.format(elapsed))

# use Accuracy as the evaluation metric
metric = Accuracy()
for batch in test_data:
    data = batch.data[0].as_in_context(ctx)
    label = batch.label[0].as_in_context(ctx)
    output = None
    for x in data:
        output = model(data.astype('float32', copy=False))
    metric.update(label, output)
print('validation %s=%f' % metric.get())
