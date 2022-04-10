#! /bin/bash

# export GENSIM_VERSION=4.0.0
# DOCKER_BUILDKIT=1 docker build -t fse-$GENSIM_VERSION --build-arg gensim=$GENSIM_VERSION .
# docker run --rm "fse-$GENSIM_VERSION"

# export GENSIM_VERSION=4.0.1
# DOCKER_BUILDKIT=1 docker build -t fse-$GENSIM_VERSION --build-arg gensim=$GENSIM_VERSION .
# docker run --rm "fse-$GENSIM_VERSION"

# export GENSIM_VERSION=4.1.0
# DOCKER_BUILDKIT=1 docker build -t fse-$GENSIM_VERSION --build-arg gensim=$GENSIM_VERSION .
# docker run --rm "fse-$GENSIM_VERSION"

# export GENSIM_VERSION=4.1.1
# DOCKER_BUILDKIT=1 docker build -t fse-$GENSIM_VERSION --build-arg gensim=$GENSIM_VERSION .
# docker run --rm "fse-$GENSIM_VERSION"

export GENSIM_VERSION=4.1.2
DOCKER_BUILDKIT=1 docker build -t fse-$GENSIM_VERSION --build-arg gensim=$GENSIM_VERSION .
docker run --rm "fse-$GENSIM_VERSION"