ARG BASE_IMAGE=python:3.11.8
FROM $BASE_IMAGE as runtime-environment

# install essential libraries
RUN apt-get update && apt-get install -y build-essential

# install project requirements
COPY env/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# install AutoGluon
RUN pip install autogluon --no-cache

# # prepare conda environment
# RUN mkdir -p /opt/conda 
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O /opt/conda/miniconda.sh \ && bash /opt/conda/miniconda.sh -b -p /opt/miniconda
# COPY /env/PJA-ASI-12C-GR2.yaml /tmp/environment.yaml
# RUN conda env create -f /tmp/PJA-ASI-12C-GR2.yaml
# RUN conda clean --all -f -y
# RUN conda activate PJA-ASI-12C-GR2
# RUN rm -f /tmp/PJA-ASI-12C-GR2.txt

# # add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy project
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .

EXPOSE 8888

# run project
CMD ["kedro", "run"]
