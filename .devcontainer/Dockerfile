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

# Install additional python packages (only for development)
RUN pip3 install sympy matplotlib numexpr 
RUN pip3 install RUN pip3 install pandas



