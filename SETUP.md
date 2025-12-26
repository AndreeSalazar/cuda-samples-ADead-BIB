# 🔧 Setup CUDA para ADead-BIB

## Tu Sistema

```
GPU: NVIDIA GeForce RTX 3060
VRAM: 12 GB GDDR6
Driver: 581.80
CUDA Version (Driver): 13.0
```

## Instalación de CUDA Toolkit

### Paso 1: Descargar CUDA Toolkit

1. Ve a: https://developer.nvidia.com/cuda-downloads
2. Selecciona:
   - Operating System: **Windows**
   - Architecture: **x86_64**
   - Version: **11** (o la más reciente)
   - Installer Type: **exe (local)**

### Paso 2: Instalar

```powershell
# Ejecutar el instalador descargado
# Seleccionar "Custom" y marcar:
# - CUDA Toolkit
# - CUDA Samples
# - Visual Studio Integration
```

### Paso 3: Verificar Instalación

```powershell
# Después de instalar, reiniciar terminal y ejecutar:
nvcc --version

# Debería mostrar algo como:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 13.0, V13.0.xxx
```

### Paso 4: Compilar Samples

```powershell
cd C:\Users\andre\OneDrive\Documentos\ADead-BIB\CUDA

# Crear directorio de build
mkdir build
cd build

# Configurar con CMake
cmake ..

# Compilar
cmake --build . --config Release
```

---

## Alternativa: Usar CUDA sin Toolkit

Si no quieres instalar el Toolkit completo, puedes usar:

### Opción A: PyTorch/CuPy (Python)

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cupy-cuda12x
```

### Opción B: Rust CUDA (para ADead-BIB)

```powershell
# Añadir a Cargo.toml
# [dependencies]
# cuda-runtime-sys = "0.3"
# cust = "0.3"
```

---

## Verificar GPU Funciona

```powershell
# Ya verificado - tu GPU está activa:
nvidia-smi

# Output:
# NVIDIA GeForce RTX 3060
# 12288MiB VRAM
# CUDA 13.0
```

---

## Próximos Pasos

1. **Instalar CUDA Toolkit** (recomendado para compilar samples)
2. **Compilar vectorAdd** como primer test
3. **Integrar con ADead-BIB** usando FFI

---

*Una vez instalado CUDA Toolkit, podremos compilar los samples y crear la integración ADead-BIB + CUDA*
