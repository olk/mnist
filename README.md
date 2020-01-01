# MNIST performance tests

The projects tests the performance of MXNet and Tensorflow using MNIST data.


## project structure

    ├── LICENSE
    ├── README.md                  <- Top-level README for developers using this project
    ├── mnist_mx_gluon.py          <- MXNet + Gluon
    ├── mnist_mx_gluon_mgpu.py     <- MXNet + Gluon + multi-gpu
    ├── mnist_mx_gluon_hvd.py      <- MXNet + Gluon + Horovod
    ├── mnist_mx_keras.py          <- MXNet + Keras
    ├── mnist_mx_keras_mgpu.py     <- MXNet + Keras + multi-gpu (multi_gpu_model())
    ├── mnist_mx_sym.py            <- MXNet + symbol/module API
    ├── mnist_mx_sym_mgpu.py       <- MXNet + symbol/module API + multi-gpu
    ├── mnist_tf_keras.py          <- Tensorflow + Keras
    └── mnist_tf_keras_mirrored.py <- Tensorflow + Keras + multi-gpu (MirroredStrategy())


### MXNet Info
    Version      : 1.5.1
    Directory    : /home/graemer/Projekte/MXNet/apache-mxnet-src-1.5.1-incubating/python/mxnet
    Num GPUs     : 2

### System Info
    system       : Linux
    node         : e5lx
    release      : 5.4.3-arch1-1
    version      : #1 SMP PREEMPT Fri, 13 Dec 2019 09:39:02 +0000

### Hardware Info
    Architektur:                     x86_64
    CPU Operationsmodus:             32-bit, 64-bit
    Byte-Reihenfolge:                Little Endian
    Adressgrößen:                    46 bits physical, 48 bits virtual
    CPU(s):                          32
    Liste der Online-CPU(s):         0-31
    Thread(s) pro Kern:              2
    Kern(e) pro Socket:              8
    Sockel:                          2
    NUMA-Knoten:                     2
    Anbieterkennung:                 GenuineIntel
    Prozessorfamilie:                6
    Modell:                          79
    Modellname:                      Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
    Stepping:                        1
    CPU MHz:                         1197.887
    Maximale Taktfrequenz der CPU:   3000,0000
    Minimale Taktfrequenz der CPU:   1200,0000
    BogoMIPS:                        4192.42
    Virtualisierung:                 VT-x
    L1d Cache:                       512 KiB
    L1i Cache:                       512 KiB
    L2 Cache:                        4 MiB
    L3 Cache:                        40 MiB
    NUMA-Knoten0 CPU(s):             0-7,16-23
    NUMA-Knoten1 CPU(s):             8-15,24-31


## test results

multigpu [[Horovod]] (2 GPUs): `horovodrun -np 2 -H localhost:2 python mnist_mx_gluon_hvd.py`

 | framework                 | duration | accuracy |
 |---------------------------|----------|----------|
 | mnist_mx_gluon.py         | 14.2s    | 0.9926   |
 | mnist_mx_keras.py         | 16.4s    | 0.9931   |
 | mnist_mx_sym.py           | 16.8s    | 0.9924   |
 | mnist_tf_keras.py         | 17.6s    | 0.9932   |
 | mnist_mx_gluon_ds.py      | 17.6s    | 0.9902   |
 |                           |          |          |
 | mnist_mx_gluon_mgpu.py    | 7.2s     | 0.9926   |
 | mnist_mx_gluon_hvd.py     | 9.0s     | 0.9963   |
 | mnist_mx_sym_mgpu.py      | 10.9s    | 0.9923   |
 | mnist_mx_keras_mgpu.py    | 11.4s    | 0.9927   |
 | mnist_tf_keras_mgpu.py    | 12.7s    | 0.9921   |
 | mnist_mx_gluon_ds_mgpu.py | 15.2s    | 0.9906   |
