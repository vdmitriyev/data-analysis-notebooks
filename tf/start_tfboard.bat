@echo off
SET PATH=C:\Soft\Anaconda3;C:\Soft\Anaconda3\Scripts;C:\Soft\Anaconda3\Library\bin;%PATH%
call activate tensorflow
tensorboard --host=localhost --port=8008 --logdir=tf-logs/nn_logs/

