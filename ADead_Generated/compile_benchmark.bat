@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%
echo Compiling benchmark v2.0...
nvcc adead_benchmark.cu -o benchmark_v2.exe -O3
echo Done!
pause
