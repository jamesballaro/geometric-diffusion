#!/bin/bash

set -e

export VFX_LIB_PATH=/opt/vfx_py

echo "Starting installation of VFX dependencies."

apt-get update
apt-get -y upgrade

echo "Starting installation of IMath."
#IMath
cd /workspace
apt-get install git cmake build-essential -y
git clone https://github.com/AcademySoftwareFoundation/Imath.git
mkdir -p /workspace/Imath/build
cd /workspace/Imath/build
cmake ..
make -j$(nproc)
make install
echo "Completed installation of IMath."

#OpenEXR
echo "Starting installation of OpenEXR."
cd /workspace
apt-get install git cmake build-essential libz-dev libdeflate-dev -y
git clone https://github.com/AcademySoftwareFoundation/openexr.git
mkdir -p /workspace/openexr/build
cd /workspace/openexr/build
cmake -DCMAKE_INSTALL_PREFIX=$VFX_LIB_PATH ..
make -j$(nproc)
make install
ldconfig
echo "Completed installation of OpenEXR."

#LibTIFF
echo "Starting installation of LibTIFF."
apt-get install libtiff-dev -y
echo "Completed installation of LibTIFF."

#OCIO
echo "Starting installation of OCIO."
cd /workspace
git clone https://github.com/AcademySoftwareFoundation/OpenColorIO.git
mkdir -p /workspace/OpenColorIO/build
cd /workspace/OpenColorIO/build
cmake .. -DCMAKE_INSTALL_PREFIX=$VFX_LIB_PATH -DOCIO_BUILD_TESTS=OFF
make -j$(nproc)
make install
pip install opencolorio==2.4.2
echo "Completed installation of OCIO."

#fmtlib
echo "Starting installation of fmtlib."
apt-get install libfmt-dev -y
echo "Completed installation of fmtlib."

#pybind11
echo "Starting installation of pybind11."
pip install "pybind11[global]"
echo "Completed installation of pybind11."

#OIIO
echo "Starting installation of OIIO."
cd /workspace
git clone https://github.com/OpenImageIO/oiio.git
mkdir -p /workspace/oiio/build
cd /workspace/oiio/build
cmake .. -DCMAKE_INSTALL_PREFIX=$VFX_LIB_PATH -DUSE_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python3) -DOpenImageIO_BUILD_MISSING_DEPS=all
make -j$(nproc)
make install
echo "Completed installation of OIIO."


export PYTHON_VERSION=`python -c 'import sys; version=sys.version_info[:2]; print(f"{version[0]}.
