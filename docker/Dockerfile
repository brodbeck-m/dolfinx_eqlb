# Import dolfinX-Image
FROM dolfinx/dolfinx:v0.6.0-r1

# --- Modify docker container ---
# Update docker container
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade &&\
    apt-get -y install vim &&\
    apt-get -y install libboost-program-options-dev

# Add bash_profil
RUN touch "/root/.bash_aliases"
RUN echo "alias ..='cd ..'" >> "/root/.bash_aliases"

# Correct include of eigen in pybind11
ARG PYBIND_EIGEN_PATH="/usr/local/lib/python3.10/dist-packages/pybind11/include/pybind11/eigen.h"
RUN sed -i 's%<Eigen/Core>%<eigen3/Eigen/Core>%' ${PYBIND_EIGEN_PATH} && \
    sed -i 's%<Eigen/SparseCore>%<eigen3/Eigen/SparseCore>%' ${PYBIND_EIGEN_PATH}

# Some additional python packages
RUN pip3 install matplotlib numexpr sympy

# --- Install dolfinx_eqlb ---
RUN git clone https://github.com/brodbeck-m/dolfinx_eqlb.git &&\
    cd /root/dolfinx_eqlb &&\
    cmake -G Ninja -B build-dir -DCMAKE_BUILD_TYPE=Release cpp/ &&\
    ninja -C build-dir install &&\
    pip3 install python/ -v --upgrade

