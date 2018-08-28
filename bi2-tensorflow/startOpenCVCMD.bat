@echo off
SET PATH=C:\Soft\Anaconda3;C:\Soft\Anaconda3\Scripts;C:\Soft\Anaconda3\Library\bin;%PATH%
REM 
REM # INSTALL Visual Studio with C++ compiler first !
REM 
REM conda create -n opencv python=3.6 anaconda
REM pip install -r requirements.txt
REM conda install --yes --file requirements.txt
REM set BOOST_ROOT=c:\Soft\Boost\boost_1_67_0
REM set BOOST_LIBRARYDIR=c:\Soft\Boost\boost_1_67_0\stage\lib
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat"
call activate opencv
cmd
REM python -m ipykernel install --user --name tensorflow --display-name "Python (tensorflow)"
REM jupyter notebook