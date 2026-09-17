#!/bin/bash

# Remove c++ build
rm -rf ./build
rm -rf ./build-dir

# Remove binding to python
rm -rf ./python/build
rm -rf ./python/dolfinx_eqlb.egg-info

# Remove pycache
rm -rf ./python/.pytest_cache
rm -rf ./python/test/unit/__pycache__
