Fit generator
=============
broxoli@DESKTOP-19319V3 ~/humpback-whale> python train.py -m cnn_mobilenet -d "dataset/train_preprocessed" -c 1024 -b 4 --image_cols Image --label_col Id
Using TensorFlow backend.
2019-03-20 20:20:39.253537: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-03-20 20:20:39.275731: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Epoch 1/1
68/68 [==============================] - 115s 2s/step - loss: 9.1284 - categorical_accuracy: 0.3345

                Result Dataframe:               Image  Id  Prediction  Match
0   0016b897a-6.jpg   1           2      0
1   0001f9222-9.jpg   6           2      0
2  00050a15a-22.jpg   2           2      1
3  0016b897a-21.jpg   1           2      0
4  0001f9222-12.jpg   6           2      0
5  00029d126-25.jpg   5           2      0
6  000f0f2bf-22.jpg   2           2      1
7  00029d126-12.jpg   5           2      0
                Total predictions: 8
                Correct predictions: 2
                Wrong predictions: 6
                Accuracy: 25.0

Train on batch
==============
broxoli@DESKTOP-19319V3 ~/humpback-whale> python train.py -m cnn_mobilenet -d "dataset/train_preprocessed" -c 1024 -b 4 --image_cols Image --label_col Id
Using TensorFlow backend.
2019-03-20 20:12:56.413680: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2019-03-20 20:12:56.430278: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
Processing Epoch: 1/1 Batch: 68/68 Loss: 4.0323686599731445: 100%|████████████████████████████████████████████████████████████████████| 1/1 [04:52<00:00, 292.22s/it]

                Result Dataframe:               Image  Id  Prediction  Match
0   0005c1ef8-8.jpg   2           2      1
1   001c1ac5f-8.jpg   4           2      0
2  000a6daec-13.jpg   0           2      0
3  0016b897a-18.jpg   1           2      0
4  0001f9222-26.jpg   6           2      0
5  0005c1ef8-18.jpg   2           2      1
6   0005c1ef8-7.jpg   2           2      1
7  001c1ac5f-14.jpg   4           2      0
                Total predictions: 8
                Correct predictions: 3
                Wrong predictions: 5
                Accuracy: 37.5

Fit
===
Epoch 1/1
1/1 [==============================] - 0s 367ms/step - loss: 3.7650 - categorical_accuracy: 0.0000e+00
Processing Epoch: 1/1 Batch: 68/68 Loss: [5.050541877746582]: 100%|███████████████████████████████████████████████████████████████████| 1/1 [02:23<00:00, 143.76s/it]

                Result Dataframe:               Image  Id  Prediction  Match
0   0016b897a-6.jpg   1           2      0
1   0001f9222-9.jpg   6           4      0
2  00050a15a-22.jpg   2           4      0
3  0016b897a-21.jpg   1           4      0
4  0001f9222-12.jpg   6           4      0
5  00029d126-25.jpg   5           4      0
6  000f0f2bf-22.jpg   2           4      0
7  00029d126-12.jpg   5           4      0
                Total predictions: 8
                Correct predictions: 0
                Wrong predictions: 8
                Accuracy: 0.0

[Colab] Batch train
===================
/content/humpback-whale
Using TensorFlow backend.
/usr/local/lib/python3.6/dist-packages/numpy/lib/arraysetops.py:472: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
  mask |= (ar1 == a)
Downloading: /run_data/densenet_trial/siamese_network_mobilenet.batch.0.epoch.0.h5: 100% 27359768/27359768 [00:01<00:00, 24868182.15it/s]
2019-03-21 01:09:35.685422: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2299995000 Hz
2019-03-21 01:09:35.685720: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x3b73fa0 executing computations on platform Host. Devices:
2019-03-21 01:09:35.685766: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-03-21 01:09:35.803926: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:998] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-03-21 01:09:35.804536: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x3b74520 executing computations on platform CUDA. Devices:
2019-03-21 01:09:35.804578: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
2019-03-21 01:09:35.804991: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:00:04.0
totalMemory: 11.17GiB freeMemory: 11.10GiB
2019-03-21 01:09:35.805036: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0
2019-03-21 01:09:36.184447: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-21 01:09:36.184512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 
2019-03-21 01:09:36.184538: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N 
2019-03-21 01:09:36.184814: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:42] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
2019-03-21 01:09:36.184877: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 10754 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0, compute capability: 3.7)
Processing Epoch: 1/1 Batch: 1/168919 Loss: None:   0% 0/1 [00:02<?, ?it/s]2019-03-21 01:10:32.463030: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
Processing Epoch: 1/1 Batch: 1000/168919 Loss: 0.9594995379447937:   0% 0/1 [21:33<?, ?it/s]
Uploading: /run_data/densenet_trial/siamese_network_mobilenet.batch.999.epoch.0.h5:   0% 0/27359768 [00:00<?, ?it/s]
Uploading: /run_data/densenet_trial/siamese_network_mobilenet.batch.999.epoch.0.h5:  31% 8388608/27359768 [00:00<00:02, 8599436.80it/s]
Uploading: /run_data/densenet_trial/siamese_network_mobilenet.batch.999.epoch.0.h5:  46% 12582912/27359768 [00:03<00:03, 4015917.14it/s]
Uploading: /run_data/densenet_trial/siamese_network_mobilenet.batch.999.epoch.0.h5:  61% 16777216/27359768 [00:04<00:02, 4009785.99it/s]
Uploading: /run_data/densenet_trial/siamese_network_mobilenet.batch.999.epoch.0.h5:  77% 20971520/27359768 [00:05<00:01, 3629868.16it/s]
Uploading: /run_data/densenet_trial/siamese_network_mobilenet.batch.999.epoch.0.h5:  92% 25165824/27359768 [00:06<00:00, 3687192.26it/s]
Uploading: /run_data/densenet_trial/siamese_network_mobilenet.batch.999.epoch.0.h5: 100% 27359768/27359768 [00:07<00:00, 2947424.86it/s]
Processing Epoch: 1/1 Batch: 1197/168919 Loss: 0.5489671230316162:   0% 0/1 [26:01<?, ?it/s]

[Colab] Fit generator
======================
Epoch 1/1
2019-03-21 01:38:24.752766: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
  1007/168919 [..............................] - ETA: 51:21:48 - loss: 0.6618 - acc: 0.6117Traceback (most recent call last):
    File "train.py", line 254, in <module>
Total: 18 minutes
