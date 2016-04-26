FROM yandex/rep-base:0.6.1
MAINTAINER Andrey Ustyuzhanin <anaderi@yandex-team.ru>

COPY scripts/_start_jupyter.sh /root/start.sh
RUN chmod +x /root/start.sh
COPY environment.yml /root/
RUN /root/miniconda/bin/conda env update -n=py27 -f=/root/environment.yml #  -q QUIET

RUN wget https://github.com/github/git-lfs/releases/download/v1.2.0/git-lfs-linux-amd64-1.2.0.tar.gz && \
    tar xzf git-lfs-linux-amd64-1.2.0.tar.gz && \
    cd git-lfs-1.2.0 && ./install.sh && git lfs install && \
    cd ../ && rm -rf git-lfs-1.2.0*
