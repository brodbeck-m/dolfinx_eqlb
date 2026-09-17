#!/bin/bash

# # --- Version using dolfinx v0.6.0 ---
# #-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON Build c++ backend
# cmake -G Ninja -B build-dir -DCMAKE_BUILD_TYPE=Release cpp/
# ninja -C build-dir install

# # Link c++ and python
# pip3 install python/ -v --upgrade

--- Version using dolfinx v0.9.0 ---
#-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON Build c++ backend
# cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-real -DCMAKE_BUILD_TYPE=Release -B build-dir cpp/
cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo  -B build-dir cpp/
ninja -j4 -C build-dir install

# Link c++ and python
python3 -m pip install -v --upgrade --no-build-isolation --check-build-dependencies \
  --config-settings=cmake.build-type=Debug \
  --config-settings=cmake.args="-DCMAKE_CXX_FLAGS='-g -O0'" \
  --target /usr/local/dolfinx-real/lib/python3.12/dist-packages \
  --no-dependencies --no-cache-dir ./python