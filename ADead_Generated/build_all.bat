@echo off
REM ============================================
REM ADead-BIB + CUDA Build Script v2.0
REM Para RTX 3060 12GB
REM Instrumentación correcta con cudaEvent
REM ============================================

echo.
echo ============================================
echo   ADead-BIB + CUDA Build System v2.0
echo   RTX 3060 12GB
echo ============================================
echo.

REM Configurar Visual Studio
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo ERROR: Visual Studio Build Tools not found
    echo Please install Visual Studio Build Tools 2022
    pause
    exit /b 1
)

REM Configurar CUDA
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set PATH=%CUDA_PATH%\bin;%PATH%

echo.
echo CUDA Version:
nvcc --version
echo.

REM Compilar todos los archivos
echo ============================================
echo Compiling CUDA files...
echo ============================================

if exist adead_benchmark.cu (
    echo [1/3] Compiling benchmark v2.0 (cudaEvent)...
    nvcc adead_benchmark.cu -o benchmark_v2.exe -O3
    if errorlevel 1 (
        echo ERROR: Failed to compile benchmark
    ) else (
        echo SUCCESS: benchmark_v2.exe created
    )
)

if exist adead_vectoradd.cu (
    echo [2/3] Compiling vectoradd.cu...
    nvcc adead_vectoradd.cu -o vectoradd.exe -O3
    if errorlevel 1 (
        echo ERROR: Failed to compile vectoradd
    ) else (
        echo SUCCESS: vectoradd.exe created
    )
)

if exist adead_matmul.cu (
    echo [3/3] Compiling matmul.cu...
    nvcc adead_matmul.cu -o matmul.exe -O3
    if errorlevel 1 (
        echo ERROR: Failed to compile matmul
    ) else (
        echo SUCCESS: matmul.exe created
    )
)

echo.
echo ============================================
echo Build complete!
echo ============================================
echo.
echo To run benchmarks:
echo   benchmark_v2.exe - Full CPU vs GPU (instrumentacion correcta)
echo   vectoradd.exe    - Vector addition test
echo   matmul.exe       - Matrix multiplication test
echo.
pause
