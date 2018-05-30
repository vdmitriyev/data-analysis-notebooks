@echo off
SET PATH=C:\Soft\Anaconda3;C:\Soft\Anaconda3\Scripts;C:\Soft\Anaconda3\Library\bin;%PATH%
python -m http.server 7777 --bind 127.0.0.1
pause