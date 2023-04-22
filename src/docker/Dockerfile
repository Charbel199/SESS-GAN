FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

COPY ./docker/requirements.txt .
COPY ./docker/3bf863cc.pub .
COPY ./docker/7fa2af80.pub .
RUN apt-key del 7fa2af80
RUN apt-key add 3bf863cc.pub
RUN apt-key add 7fa2af80.pub



RUN apt-get update && apt-get install -y wget gnupg curl && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
CMD ["tail", "-f", "/dev/null"]