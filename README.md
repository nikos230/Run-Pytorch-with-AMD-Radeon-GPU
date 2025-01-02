# Run-Pytorch-with-AMD-Radeon-GPU

## Introducation
With this guide you will be able to run Pytorch 2.1.1 with an Radeon GPU, it has been tested on rx470 4GB. Your GPU need to belong to gfx803 family like RX400 and RX500 Series. 

### Requirements
- Ubuntu 22.04 LTS
- AMD Radeon GPU in gfx803 family (rx460, rx470, rx480, rx550, rx560, rx560, rx570, rx580)

### Download Required Files
- Download [Pytorch 2.1.1](https://drive.google.com/file/d/1Tkyqe8VxUPkpf_jLZRzJphKNlW5Cqixi/view?usp=sharing) or build it yourself (see below)
- Download [rocblas_2.46.0.50401-84.20.04_amd64.deb](https://github.com/xuhuisheng/rocm-gfx803/releases/tag/rocm541)


## Guide to Setup ROCm and Pytorch

Start with a fresh setup of ubuntu 22.04, then you need to install AMD drivers like ROCm. The version needed is ROCm 5.4.0, or 5.4.3 choose one of theese.
- Open a terminal and type
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
sudo su
sudo echo ROC_ENABLE_PRE_VEGA=1 >> /etc/environment
sudo echo HSA_OVERRIDE_GFX_VERSION=8.0.3 >> /etc/environment
</pre>
Reboot your system <br /><br />


- Open terminal and now you can start installing ROCm (for ROCm 5.4.0)

<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
cd Downloads
wget https://repo.radeon.com/amdgpu-install/5.4/ubuntu/jammy/amdgpu-install_5.4.50400-1_all.deb
sudo apt install ./amdgpu-install_5.4.50400-1_all.deb
sudo amdgpu-install -y --no-dkms --usecase=rocm,hiplibsdk,mlsdk
sudo usermod -aG video $LOGNAME
sudo usermod -aG render $LOGNAME
</pre>

- or Alternative install ROCm 5.4.3
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
cd Downloads
wget https://repo.radeon.com/amdgpu-install/5.4.3/ubuntu/jammy/amdgpu-install_5.4.50403-1_all.deb
sudo apt install ./amdgpu-install_5.4.50403-1_all.deb
sudo amdgpu-install -y --no-dkms --usecase=rocm,hiplibsdk,mlsdk
sudo usermod -aG video $LOGNAME
sudo usermod -aG render $LOGNAME
</pre>
Reboot your system<br /><br />

- Open terminal and check if ROCm is installed correctly
  
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
rocminfo
clinfo
</pre>  
<br />

 - Then install libopenmpi3 andlibstdc++-11-dev and rocblas patched version for gfx803

<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
sudo apt install libopenmpi3 libstdc++-11-dev
sudo apt-get install libopenblas-dev
cd Downloads
sudo apt install ./rocblas_2.46.0.50401-84.20.04_amd64.deb 
</pre><br /><br />  

- Now you need to install Pytorch, you can use the pre-build wheels from this repo, or you can build it yourself but it will take some time. You can not install Pytorch with ROCm support directly from the Pytorch repo because it will not work for gfx803 GPUs
  
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
cd Downloads
sudo apt install pip
pip install torch-2.1.1-cp310-cp310-linux_x86_64.whl
</pre><br /><br />  


### Build Pytorch for gfx803 (patched vesrion)
- First install Dependencies
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
sudo apt install build-essential cmake python3-dev python3-numpy ninja-build libomp-dev libcurl4-openssl-dev libgflags-dev libgoogle-glog-dev libssl-dev libyaml-cpp-dev git
</pre>
- Now you can start the build
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
git clone https://github.com/pytorch/pytorch.git -b v2.1.1
cd pytorch
export PATH=/opt/rocm/bin:$PATH ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm/hip
export PYTORCH_ROCM_ARCH=gfx803
export PYTORCH_BUILD_VERSION=2.1.1 PYTORCH_BUILD_NUMBER=1
python3 tools/amd_build/build_amd.py
USE_ROCM=1 USE_NINJA=1 python3 setup.py bdist_wheel
pip3 install dist/torch-2.1.1-cp310-cp310-linux_x86_64.whl
</pre>
if you get error from rocblas library and you are using ROCm 5.4.0 create the follwing symbolc link
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
CMake Error at /opt/rocm-5.4.0/lib/cmake/rocblas/rocblas-targets.cmake:79 (message):
  The imported target "roc::rocblas" references the file

     "/opt/rocm-5.4.0/lib/librocblas.so.0.1.50400"
  but this file does not exist.  Possible reasons include:
  * The file was deleted, renamed, or moved to another location.
  * An install or uninstall procedure did not complete successfully.
  * The installation package was faulty and contained
     "/opt/rocm-5.4.0/lib/cmake/rocblas/rocblas-targets.cmake"
  but not all the files it references.
Call Stack (most recent call first):
  /opt/rocm/lib/cmake/rocblas/rocblas-config.cmake:92 (include)
  cmake/public/LoadHIP.cmake:161 (find_package)
  cmake/public/LoadHIP.cmake:291 (find_package_and_print_version)
  cmake/Dependencies.cmake:1268 (include)
  CMakeLists.txt:722 (include)
</pre>
<pre style="background-color: #f4f4f4; padding: 10px; border-radius: 8px;">
sudo ln -s /opt/rocm-5.4.1/lib/librocblas.so.0 /opt/rocm-5.4.0/lib/librocblas.so.0
sudo ln -s /opt/rocm-5.4.1/lib/librocblas.so.0.1.50401 /opt/rocm-5.4.0/lib/librocblas.so.0.1.50400
</pre>

## References
[https://github.com/tsl0922/pytorch-gfx803](https://github.com/tsl0922/pytorch-gfx803) <br />
[https://github.com/xuhuisheng/rocm-gfx803](https://github.com/xuhuisheng/rocm-gfx803)
