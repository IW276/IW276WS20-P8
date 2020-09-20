# Project-Template for IW276 Autonome Systeme Labor

This project includes optimization of the existing methods and technologies with the aim of detecting larger numbers of people from CCTV cameras in public places.
Our approach works with the help of a pre-trained machine learning model that runs on a Jetson Nano in a Docker container.

<p align="center">
  Screenshot / GIF <br />
  Link to Demo Video
</p>

> This work was done by Stanislava Anastasova, Kristina Koleva, Tatsiana Mazouka during the IW276 Autonome Systeme Labor at the Karlsruhe University of Applied Sciences (Hochschule Karlruhe - Technik und Wirtschaft) in WS 2020 / 2021. 

## Table of Contents

* [Requirements](#requirements)
* [Prerequisites](#prerequisites)
* [Pre-trained model](#pre-trained-model)
* [Running](#running)
* [Acknowledgments](#acknowledgments)

## Requirements
* Python 3.6 (or above)
* OpenCV 4.1 (or above)
* Jetson Nano
* Jetpack 4.4
* Registration in Google Colab
> [Optional] ...

## Prerequisites
### For work with Jetson Nano: 
1. Install requirements:
```
pip install -r requirements.txt
```

### For network training:
1. Open Google Colab and register there
2. Upload the dataset for training. There are three possibilities:
  a. Upload the open-image dataset with help of this command:
  ```
  !bash python open_images_downloader.py --max-images=2500 --class-names "Person"
  ```

  b. Upload your own dataset in Colab environment via right mouse click in the colab directory area. This approach is very time consuming and the data will be deleted after the expiration of runtime. That why it is better to use the approach from the a or c list items. 

  c. Upload your dataset(with the same format as open-image dataset) in Google Drive and mount the Google Drive with the Google Colab using following comand:
  ```
  from google.colab import drive
  drive.mount('/content/drive')
  ```
3. Upload all necessary scripts in Google Colad. The most convinient way to do it is to put all necessary files in one ".zip"-File and unzip it in Colab Notebook using following comand:
  ```
  !unzip Files_for_colab.zip
  ```



## Pre-trained models <a name="pre-trained-models"/>

Pre-trained model is available at pretrained-models/

## Running

To run the training, pass path to the pre-trained checkpoint:
```
!python train_ssd.py --model-dir=/content/drive/My\ Drive/ASLabor/models/people --num-epochs=30 --data=/content/drive/My\ Drive/ASLabor/data --batch-size=4 --resume=/content/drive/My\ Drive/ASLabor/models/people/mb1-ssd-Epoch-14-Loss-6.35830452063909.pth
```
Here are some common options that you can run the training script with:

| Argument       |  Default  | Description                                                |
|----------------|:---------:|------------------------------------------------------------|
| `--data`       |  `data/`  | the location of the dataset                                |
| `--model-dir`  | `models/` | directory to output the trained model checkpoints          |
| `--resume`     |    None   | path to an existing checkpoint to resume training from     |
| `--batch-size` |     4     | try increasing depending on available memory               |
| `--epochs`     |     30    | up to 100 is desirable, but will increase training time    |
| `--workers`    |     2     | number of data loader threads (0 = disable multithreading) |

> Additional comment about the demo.

## Docker
HOW TO

## Acknowledgments

This repo is based on
  - [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md)
  - [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
