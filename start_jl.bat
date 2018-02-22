@echo off
SET PATH=C:\Soft\Anaconda3;C:\Soft\Anaconda3\Scripts;C:\Soft\Anaconda3\Library\bin;%PATH%
call activate jupyterlab
jupyter lab --port=8899 --notebook-dir=jupyterlab
REM https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906
REM jupyter notebook