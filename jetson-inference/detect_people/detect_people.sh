#!/bin/bash
cd python/training/detection/ssd

python3 onnx_export.py --model-dir=models/people

detectnet --model=models/people/ssd-mobilenet.onnx --labels=models/people/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            "images/test/*" output

rm models/people/ssd-mobilenet.onnx*
