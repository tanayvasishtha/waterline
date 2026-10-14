FROM public.ecr.aws/lambda/python:3.12

# OpenCV needs libGL at runtime even in the headless wheel
RUN dnf install -y mesa-libGL && dnf clean all

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY sites ./sites

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
        "opencv-python-headless==5.0.0.93" "numpy>=2.2" "pydantic>=2.9" \
        "httpx>=0.28" "boto3>=1.35" "boto3-stubs[bedrock-runtime]>=1.43" && \
    uv pip install --system --no-cache .

ENV WATERLINE_SITES_DIR=/var/task/sites
CMD ["waterline.aws.handler"]
