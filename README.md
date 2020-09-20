# Project-Template for IW276 Autonome Systeme Labor

This project includes the optimization of the existing methods and technologies to detect larger numbers of people from CCTV cameras in public places.
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
// TODO Sissi
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

  b. Upload your dataset in Colab environment via the right mouse click in the Colab directory area. This approach is very time consuming and the data will be deleted after the expiration of runtime. That why it is better to use the approach from the a or c list items. 

  c. Upload your dataset(with the same format as open-image dataset) in Google Drive and mount the Google Drive with the Google Colab using the following command:
  ```
  from google.colab import drive
  drive.mount('/content/drive')
  ```
3. Upload all necessary scripts in Google Colad. The most convenient way to do it is to put all necessary files in one ".zip"-File and unzip it in Colab Notebook using the following command:
  ```
  !unzip Files_for_colab.zip
  ```



## Pre-trained models <a name="pre-trained-models"/>

Pre-trained model is available at pretrained-models/

## Running

### Run data splitting
The input images from https://bwsyncandshare.kit.edu are very big and they must be split in small images for fast training. 
[Panda-Toolkit](https://github.com/IW276/PANDA-Toolkit) was used to split the input picture in a lot of small images. You can find the scripts for the data splitting in the folder "pythonsplittingProject". 
You should run following script from this folder to split the train data:
```
python PANDA-Toolkit/generate_split_data.py --image_root ./ --person_anno_file image_ann
os/image_annos/person_bbox_train.json --output_dir output_train  --image_subdir image_train  --annotype train
```
You should run following script from this folder to split the validation data:
```
python PANDA-Toolkit/generate_split_data.py --image_root ./ --person_anno_file image_annos/perso
n_bbox_valid.json --output_dir output_valid --image_subdir ./image_valid  --annotype valid
```
Here are some common options that you can run the splitting script with:

| Argument             | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `--image_root`       | the path to the root directory                                     |
| `--person_anno_file` | the path to the annotation file                                    |
| `--output_dir`       | the path to the output directory                                   |
| `--image_subdir`     | the name of the subdirectory to input images that must be splitted |
| `--annotype`         | type of the annotations                                            |

### Run format conversion for annotations 

To prepare the dataset for network training, the images and annotations must have a similar format as data from open-image dataset, but the annotations have COCO-Format after data splitting. The annotations will be converted to the correct format with help of the script in folder "ConvertCOCOtoCSV". To convert the annotations you should adjust the name of ".json"-file in code and run the main.py script. 

### Run training 
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

The train_ssd.py is taken from [jetson-inference repository](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md). The execution of the script produces the ".pth"-file, which can be run on Jetson Nano to detect the people from the photos or videos. The script is executed in Google Colab, because we need GPU to run the training fast. Here is the link to our [Colab Notebook](https://colab.research.google.com/drive/1qh2uV86M_5wnlsHCOIelaP0klH1zyBKR?usp=sharing).

### Run model on Jetson Nano

// TODO Sissi

## Docker

// TODO Sissi

## Acknowledgments

This repo is based on
  - [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md)
  - [Panda-Toolkit](https://github.com/IW276/PANDA-Toolkit)
  - [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)

Thanks to the original authors for their work!

## Contact
Please email `mickael.cormier AT iosb.fraunhofer.de` for further questions.
