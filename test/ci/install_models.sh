#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/models/$MODELS_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  cd /tmp

  mkdir -p $CACHE_DIR

  wget https://github.com/ankane/informers/releases/download/v$MODELS_VERSION/sentiment-analysis.onnx
  mv sentiment-analysis.onnx $CACHE_DIR

  wget https://github.com/ankane/informers/releases/download/v$MODELS_VERSION/question-answering.onnx
  mv question-answering.onnx $CACHE_DIR

  wget https://github.com/ankane/informers/releases/download/v$MODELS_VERSION/feature-extraction.onnx
  mv feature-extraction.onnx $CACHE_DIR
else
  echo "Models cached"
fi
