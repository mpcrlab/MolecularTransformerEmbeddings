#!/bin/bash

docker run \
    -it \
    --rm \
    -p 7777:8888 \
    -v $(pwd):/external \
    --gpus all \
    --ipc="host" \
    --name mte-notebook \
    moleculartransformerembeddings
