@echo off
REM ADead-BIB CUDA Compiler Script
REM Requires: CUDA Toolkit + Visual Studio Build Tools

echo ========================================
echo ADead-BIB + CUDA Compiler
echo ========================================

REM Setup Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
if errorlevel 1 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

REM Add CUDA to PATH
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;%PATH%

echo.
echo Compiling VectorAdd...
nvcc adead_vectoradd.cu -o vectoradd.exe
if errorlevel 1 (
    echo ERROR: Failed to compile vectoradd
) else (
    echo SUCCESS: vectoradd.exe created
)

echo.
echo Compiling MatMul...
nvcc adead_matmul.cu -o matmul.exe
if errorlevel 1 (
    echo ERROR: Failed to compile matmul
) else (
    echo SUCCESS: matmul.exe created
)

echo.
echo ========================================
echo Compilation complete!
echo ========================================
pause
