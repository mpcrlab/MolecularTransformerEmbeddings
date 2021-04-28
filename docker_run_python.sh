#!/bin/bash

docker run \
    -it \
    --rm \
    -v $(pwd):/external \
    --gpus all \
    --ipc="host" \
    --name mte-notebook \
    --entrypoint python3
    moleculartransformerembeddings
    embed.py