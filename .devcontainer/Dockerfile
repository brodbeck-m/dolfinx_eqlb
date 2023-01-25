# Initialize from local docker image
FROM dolfinx/dolfinx:latest

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

# Install additional python packages (only for development)
RUN pip3 install sympy matplotlib numexpr 
RUN pip3 install autopep8 pandas



