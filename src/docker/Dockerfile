FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel

COPY ./docker/requirements.txt .

RUN apt-get update && apt-get install -y wget gnupg curl && apt-get install --no-install-recommends -y \
    libgl1 \
    libgomp1 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

