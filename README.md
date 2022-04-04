# ResDepth: A Deep Residual Prior For 3D Reconstruction From High-resolution Satellite Images

![ResDepth](docs/teaser.png?raw=true)

This repository provides the code to train and evaluate ResDepth, an efficient and easy-to-use neural architecture for 
learned DSM refinement from satellite imagery. It represents the official implementation of the paper:

### [ResDepth: A Deep Residual Prior For 3D Reconstruction From High-resolution Satellite Images](https://doi.org/10.1016/j.isprsjprs.2021.11.009)
*[Corinne Stucker](https://prs.igp.ethz.ch/content/specialinterest/baug/institute-igp/photogrammetry-and-remote-sensing/en/group/people/person-detail.html?persid=179102), 
[Konrad Schindler](https://prs.igp.ethz.ch/group/people/person-detail.schindler.html)*

> **Abstract:** *Modern optical satellite sensors enable high-resolution stereo reconstruction from space. But the challenging imaging conditions when observing the Earth from space push stereo matching to its limits. In practice, the resulting digital surface models (DSMs) are fairly noisy and often do not attain the accuracy needed for high-resolution applications such as 3D city modeling. Arguably, stereo correspondence based on low-level image similarity is insufficient and should be complemented with a-priori knowledge about the expected surface geometry beyond basic local smoothness. To that end, we introduce ResDepth, a convolutional neural network that learns such an expressive geometric prior from example data. ResDepth refines an initial, raw stereo DSM while conditioning the refinement on the images. I.e., it acts as a smart, learned post-processing filter and can seamlessly complement any stereo matching pipeline. In a series of experiments, we find that the proposed method consistently improves stereo DSMs both quantitatively and qualitatively. We show that the prior encoded in the network weights captures meaningful geometric characteristics of urban design, which also generalize across different districts and even from one city to another. Moreover, we demonstrate that, by training on a variety of stereo pairs, ResDepth can acquire a sufficient degree of invariance against variations in imaging conditions and acquisition geometry.*


## Requirements

This code has been developed and tested on Ubuntu 18.04 with Python 3.7, PyTorch 1.9, and GDAL 2.2.3. 
It may work for other setups but has not been tested thoroughly.

On Ubuntu 18.04, `gdal` can be installed with `apt-get`:
```shell
sudo apt update
sudo apt install libgdal-dev gdal-bin
```


## Setup

Before proceeding, make sure that GDAL is installed and set up correctly.

To create a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) and install the required 
dependencies, please run:
```bash
git clone https://github.com/stuckerc/ResDepth.git
cd ResDepth
python3 -m venv tmp/resdepth
source tmp/resdepth/bin/activate
(resdepth) $ pip install --upgrade pip setuptools wheel
(resdepth) $ pip install -r requirements.txt
```
in your working directory. Next, use the following command to install GDAL in the previously created virtual environment:
```bash
(resdepth) $ pip install --global-option=build_ext --global-option="-I/usr/include/gdal" GDAL==`gdal-config --version`
```


## Quick Start

ResDepth follows a residual learning strategy, i.e., it is trained to refine an imperfect input DSM by regressing a 
per-pixel correction to the height, using both the DSM and ortho-rectified panchromatic (stereo) images as input. 


### Data Preparation

We assume that the initial surface reconstruction has been generated with existing multi-view stereo matching and/or 
depth map fusion techniques. Furthermore, we assume that the images have already been ortho-rectified with the help of 
the initial surface estimate. Please note that this repository does not provide any functionality to perform data 
pre-processing (initial surface reconstruction, ortho-rectification) nor image pair selection.

When preparing your data as input for ResDepth, make sure to meet the following requirements:
* The initial reconstruction and the reference model are parametrized as digital surface models (DSMs, i.e., height 
fields in raster format).
* The initial DSM and ground truth DSM are reprojected to the same horizontal coordinate system (for instance, the 
local UTM zone). The pixel values of the initial DSM and the ground truth DSM must refer to the same vertical coordinate 
system or otherwise be vertically aligned.
* ResDepth accepts ground truth DSMs that exhibit NoData cells. However, NoData cells in the initial DSM are prohibited. 
Hence, you might have to interpolate the initial DSM to replace any NoData cells.  
* The initial DSM is used to ortho-rectify the panchromatic satellite views.
* The initial DSM, the ground truth DSM, and the ortho-images are stored in GeoTIFF raster format. Furthermore, all 
rasters must be co-registered, cropped to the same spatial (rectangular) extent, and exhibit the same spatial resolution 
(i.e., pixel-aligned rasters).  

>  **Note**: Throughout our experiments, we use DSMs with a grid spacing of 0.25 m. By construction, ResDepth is 
>generic and can be trained to refine any DSM. However, if the spatial resolution deviates from our setting, it might be 
>required to adapt the tile size ```tile_size``` of the DSM patches and/or the depth ```depth``` of the U-Net  
>(number of downsampling and upsampling layers). 


For evaluation, ResDepth accepts the following additional rasters as input:
* *Ground truth mask*: A raster in GeoTIFF format. A pixel value of 1 indicates a pixel with valid reference height, 
whereas a pixel value of 0 indicates a pixel with invalid reference height. This raster is used to mask out areas with 
evident temporal differences between the initial and ground truth DSM due to construction activities.
* *Building mask*: A raster in GeoTIFF format. A pixel with a value of 1 indicates a building pixel, and a pixel with a 
value of 0 is a terrain pixel. If a building mask is provided, the quantitative errors are computed for buildings and 
terrain separately. Before calculating the object-specific metrics, the building mask is dilated by two pixels to avoid 
aliasing at vertical walls.
* *Water mask*: A raster in GeoTIFF format. A pixel with a value of 1 indicates a water pixel, and a pixel with a value 
of 0 is a non-water pixel. Water bodies are excluded from the evaluation only if a building mask is provided.  
* *Forest mask*: A raster in GeoTIFF format. A pixel with a value of 1 indicates a forest pixel, and a pixel with a value 
of 0 is a non-forest pixel. Densely forested areas are excluded from the evaluation only if a building mask is provided.

> **Note**: All rasters (DSMs, ortho-images, mask rasters) must be co-registered and cropped to the same spatial 
>(rectangular) extent. Furthermore, the spatial resolution must be the same.


### Data Structure

We do not expect a particular structure of how the data is stored. The path to every initial DSM, to the corresponding 
ground truth DSM, and possibly to raster masks must be listed in the configuration file (see below). The ortho-images 
and the definition of the image pairs must be provided as text files.


#### Image List

Prepare a text file ```imagelist.txt``` that lists the absolute paths to the pre-computed ortho-rectified satellite 
images (one file path per line):
```shell
path/to/ortho-image1.tif
path/to/ortho-image2.tif
path/to/ortho-image3.tif
path/to/ortho-image4.tif
path/to/ortho-image5.tif
path/to/ortho-image6.tif
path/to/ortho-image7.tif
...
```

It is possible to use the same image list for training, validation, and testing.

> **Note**: The filename of every image listed in the image list has to be unique (irrespective of the absolute file 
>path due to the definition of the image pair list, see below).


#### Image Pair List

Prepare a text file ```pairlist.txt``` that comprises a comma-separated list of filenames, where every line defines one 
image pair. If multiple image pairs are specified, each pair needs to be of equal length (i.e., the same number of  
images per image pair).

***Example image pair list for ResDepth-stereo***: The following image pair list defines a single stereo pair composed 
of the images ```ortho-image1.tif``` and ```ortho-image2.tif```. The absolute paths to the ortho-images will be derived 
by matching the image filenames listed in ```pairlist.txt``` and ```imagelist.txt```.
```shell
ortho-image1.tif, ortho-image2.tif
```

***Example image pair list for ResDepth-stereo, generalized across viewpoints***: To train a ResDepth-stereo network 
that generalizes across variations in acquisition geometry and the images' radiometry, one simply has to provide 
multiple stereo pairs in the image pair list, for example:
```shell
ortho-image1.tif, ortho-image2.tif
ortho-image2.tif, ortho-image5.tif
ortho-image4.tif, ortho-image5.tif
```

In this example, ResDepth-stereo will be trained using the three image pairs (```ortho-image1.tif```, ```ortho-image2.tif```), 
(```ortho-image2.tif```, ```ortho-image5.tif```), and (```ortho-image4.tif```, ```ortho-image5.tif```).


***Example image pair list for ResDepth-mono***: The following example shows an image pair list to train ResDepth-mono  
using the single image ```ortho-image1.tif``` as guidance:
```shell
ortho-image1.tif, 
```

***Example image pair list for ResDepth-0***: The network variant ResDepth-0 does not leverage any satellite views. 
Therefore, to train or evaluate ResDepth-0, neither the image list ```imagelist.txt``` nor the image pair list 
```pairlist.txt``` have to be provided.


## Training

To train ResDepth, run the script ```train.py``` with a JSON configuration file as the unique argument:
```shell
(resdepth) $ python train.py config.json
```

The configuration file ```config.json``` specifies the input data and the output directory (see below for details). 
Furthermore, it configures the model architecture, hyperparameters, and training settings. All parameters and their 
default settings are described in ```./lib/config.py```.

To monitor and visualize the training process, you can start a tensorboard session with:
```shell
(resdepth) $ tensorboard --logdir <tboard_log_dir>
```


## Evaluation

To evaluate the ResDepth prior, run the script ```test.py``` with a JSON configuration file as the unique argument:
```shell
(resdepth) $ python test.py config_test.json
```

The configuration file ```config_test.json``` specifies the input data, the model architecture of ResDepth and its 
weights, and the output directory (see below for details).

The script ```test.py``` uses a tiling-based strategy to refine the given input DSM. First, it cuts the DSM into 
a regular grid of overlapping tiles, where the tile size amount to ```tile_size``` and the stride to ```0.5*tile_size```. 
The DSM patches are then individually refined. Lastly, the refined DSM patches are merged to output a single refined 
DSM raster.
If multiple images (ResDepth-mono) or image pairs (ResDepth-stereo) are provided in the image pair list, the same 
initial DSM is refined multiple times by using every image (pair) once for guidance. Finally, the error metrics are 
reported both over all predictions and for every prediction separately.


## Configuration File: Training

The following example shows the bare minimum JSON configuration file ```config.json``` to train ResDepth. It consists of 
two objects ```datasets``` and ```output``` with mandatory and optional name-value pairs that need to be completed by 
the user. 
```json
{
  "datasets": [
    {
      "name": "my_dataset",
      "raster_gt": "path/to/ground_truth_DSM.tif",
      "raster_in": "path/to/initial_DSM.tif",
      "path_image_list": "path/to/imagelist.txt",
      "path_pairlist_training": "path/to/pairlist_training.txt",
      "path_pairlist_validation": "path/to/pairlist_validation.txt",
      "area_type": "train+val",
      "allocation_strategy": "5-crossval_vertical",
      "test_stripe": 1,
      "crossval_training": false,
      "n_training_samples": 20000
    }
    ],
  "output": {
    "suffix": "",
    "output_directory": "path/to/output_directory",
    "tboard_log_dir": "path/to/tboard_log_directory"
  }
}
```

### Input Data

The ```datasets``` object defines a list of objects with mandatory and optional key-value pairs. Every object in the list 
describes a dataset, i.e., a (rectangular) geographic region for which an initial DSM, a corresponding ground truth DSM, 
and ortho-images are available. For training, ResDepth expects at least one training dataset and one validation dataset 
(i.e., the list is composed of two objects). Alternatively, ResDepth accepts one (or multiple) dataset(s) split into 
mutually exclusive stripes for training, validation, and testing (i.e., the list is composed of at least one object). 
Every dataset (object in the list) has the following mandatory key-value pairs:

* ```raster_in```: str, path to the initial DSM.
* ```raster_gt```: str, path to the corresponding ground truth DSM.
* ```path_image_list```: str, path to the image list ```imagelist.txt```.
* ```path_pairlist_training```: str, path to the image pair list ```pairlist_training.txt```, which defines the image 
pairs used for training.
* ```path_pairlist_validation```: str, path to the image pair list ```pairlist_validation.txt```, which defines the image 
pairs used for validation (the image pairs used for training and validation can be equal or different from each other).
* ```area_type```: str, choose among ['train', 'val', 'train+val']. The parameter ```area_type``` specifies whether the 
dataset/region is used for training ('train'), for validation ('val'), or for both ('train+val').

The keys ```path_image_list```, ```path_pairlist_training```, and ```path_pairlist_validation``` are not required for 
ResDepth-0.

Additionally, the user can specify the following optional key-value pairs:
* ```name```: str, optional dataset identifier.
* ```allocation_strategy```: str, choose among ['5-crossval_vertical', '5-crossval_horizontal', 'entire']. If the parameter 
```allocation_strategy``` is set to 'entire', the entire dataset/region will be used for either training or validation 
(depending on the value of ```area_type```). Set the parameter ```allocation_strategy``` to '5-crossval_vertical' to 
split the geographic region into five equally large and mutually exclusive vertical stripes (north-south oriented). 
Similarly, set ```allocation_strategy``` to '5-crossval_horizontal' to split the geographic region into five equally 
large and mutually exclusive horizontal stripes (west-east oriented).
* ```test_stripe```: int, choose among [0, 1, 2, 3, 4]. The parameter ```test_stripe``` is relevant if the value of 
```allocation_strategy``` is in ['5-crossval_vertical', '5-crossval_horizontal']. It determines which of the five stripes 
is reserved for testing. By definition, the validation stripe is located to the right/bottom (east/south) of the test 
stripe (cyclic order). The remaining stripes are used for training.
* ```crossval_training```: bool, flag to activate (True) or deactivate (False) cross-validation. To perform cross-validation, 
set ```crossval_training``` to True, ```area_type``` to 'train+val', and choose among ['5-crossval_vertical', '5-crossval_horizontal'] 
to specify ```allocation_strategy```. The stripe with index ```test_stripe``` is used as validation region and all other 
stripes as training regions.
* ```n_training_samples```: int, number of DSM patches randomly sampled from the training region. This parameter  
has to be specified only if ```area_type``` is in ['train', 'train+val'].

> **Warning**: At runtime, all rasters (DSMs and ortho-images listed in ```imagelist.txt```) are loaded to memory.


### Output Settings

The name of the results folder consists of the code execution day and time and an optional suffix ```YYYY-MM-DD_HH-MM_${suffix}```. 
The ```output``` object consists of the following key-value pairs:  
* ```suffix```: str, optional suffix appended to the name of the results folder.
* ```output_directory```: str, output directory. All log files will be stored in the directory 
```${output_directory}/YYYY-MM-DD_HH-MM_${suffix}``` and the PyTorch checkpoints in 
```${output_directory}/YYYY-MM-DD_HH-MM_${suffix}/checkpoints``` (saved intermediate and final model weights). 
* ```tboard_log_dir```: str, output directory. All tensorboard checkpoint files (event files) will be stored in the 
directory ```${tboard_log_dir}/YYYY-MM-DD_HH-MM_${suffix}```.


The results directory is structured as follows:
```shell
${output_directory}/YYYY-MM-DD_HH-MM_${suffix}/
├── checkpoints
│   ├── Model_after_${save_model_rate}_epochs.pth
│   ├── Model_after_${2*save_model_rate}_epochs.pth
│   ├── ...
│   └── Model_best.pth
├── config.json
├── config.json.orig
├── DSM_normalization_parameters.p
├── Image_normalization_parameters.p
├── model_config.json
├── run.log
└── training.log
```

It stores the following files:
* ```config.json.orig```: JSON file, configuration file used to invoke ```train.py```.
* ```config.json```: validated and augmented JSON file, the combination of ```config.json.orig``` and the default settings 
specified in ```./lib/config.py```.
* ```model_config.json```: JSON file, stores the model architecture of ResDepth.
* ```DSM_normalization_parameters.p```: pickle file, stores the DSM normalization parameters.
* ```Image_normalization_parameters.p```: pickle file, stores the satellite image normalization parameters.
* ```run.log```: log file, console output.
* ```training.log```: log file, training progress.
* ```checkpoints/Model_after_${save_model_rate}_epochs.pth```: model weights after ```save_model_rate``` training epochs.
* ```checkpoints/Model_best.pth```: model weights of the training epoch with lowest validation loss.


### Switching between ResDepth-stereo, ResDepth-mono, and ResDepth-0

The content of the image pair list ```pairlist.txt``` determines whether ResDepth is trained using stereo information 
or a single image as guidance. To train ResDepth-0, neither the image list ```imagelist.txt``` nor the image pair list 
```pairlist.txt``` have to be provided.

In addition to modifying the image pair list ```pairlist.txt```, one must also adjust the network architecture 
accordingly. For ResDepth-stereo, add the following settings to your JSON configuration file:
```json
  "model": {
    "input_channels": "geom-stereo"
  },
  "stereopair_settings": {
    "use_all_stereo_pairs": true,
    "permute_images_within_pair": true
  }
```

These are the default settings to train ResDepth-stereo, which generalizes across variations in acquisition geometry and 
imaging conditions. Ideally, the image pair list specified by ```path_pairlist_training``` comprises more than one image 
pair. Set ```permute_images_within_pair``` to False if the goal is to train a ResDepth-stereo prior tailored to the 
specific image characteristics and acquisition geometry of a ***single*** image pair.

For ResDepth-mono, please specify:
```json
  "model": {
    "input_channels": "geom-mono"
  }
```

Similarly, for ResDepth-0, specify:
```json
  "model": {
    "input_channels": "geom"
  }
```

Lastly, we also provide the option to train a U-Net variant that directly regresses a DSM from an ortho-rectified stereo
pair [[1](#myfootnote1)]:
```json
  "model": {
    "input_channels": "stereo",
    "outer_skip": false
  }
```


### Changing the Default Model and Training Settings (Hyperparameters, Optimizer, Learning Rate Scheduler)

We provide a detailed description of all parameters and their default settings in ```./lib/config.py```. Most likely, the 
parameters ```depth``` and ```tile_size``` have to be fine-tuned if the spatial resolution of the DSMs deviates from 0.25 m 
(our setting). Add the parameters that you wish to modify to your JSON configuration file to overwrite the respective 
default setting.


### Templates

We provide several template files in the directory ```./configs/``` to train ResDepth-0, ResDepth-mono, and ResDepth-stereo 
on a single dataset. Furthermore, we provide a template for the generalized ResDepth-stereo variant.



## Configuration File: Evaluation

In the following, we show an example JSON configuration file ```config_test.json``` to evaluate ResDepth:
```json
{
  "datasets": [
    {
      "name": "my_dataset",
      "raster_gt": "path/to/ground_truth_DSM.tif",
      "raster_in": "path/to/initial_DSM.tif",
      "path_image_list": "path/to/imagelist.txt",
      "path_pairlist": "path/to/pairlist_test.txt",
      "mask_ground_truth": "path/to/ground_truth_mask.tif",
      "mask_building": "path/to/building_mask.tif",
      "mask_water": "path/to/water_mask.tif",
      "mask_forest": "path/to/forest_mask.tif",
      "area_type": "test",
      "allocation_strategy": "5-crossval_vertical",
      "test_stripe": 1,
      "crossval_training": false
    }
  ],
  "model": {
    "weights": "${output_directory}/YYYY-MM-DD_HH-MM_${suffix}/checkpoints/Model_best.pth",
    "architecture": "${output_directory}/YYYY-MM-DD_HH-MM_${suffix}/model_config.json",
    "normalization_geom": "${output_directory}/YYYY-MM-DD_HH-MM_${suffix}/DSM_normalization_parameters.p",
    "normalization_image": "${output_directory}/YYYY-MM-DD_HH-MM_${suffix}/Image_normalization_parameters.p"
  },
  "general": {
    "tile_size": 256
  },
  "output": {
    "directory": "path/to/results/folder"
  }
}
```

The key-value pairs of the ```datasets``` object are equal to those used for the training configuration file. 
Additionally, the user can specify the file paths of ground truth masks, building masks, water masks, and forest masks 
(see Section 'Data Preparation' above). To evaluate cross-validation, set ```crossval_training``` to True and 
```area_type``` to 'val'. Furthermore, use the same values for ```allocation_strategy``` and ```test_stripe``` as 
during training.

The ```model``` object specifies the model weights, the model architecture, and the parameters used for data normalization. 
The directory ```${output_directory}/YYYY-MM-DD_HH-MM_${suffix}``` corresponds to the output folder of the training 
script ```train.py```. Note that ```normalization_image``` is required for ResDepth-mono and ResDepth-stereo but not 
for ResDepth-0. 

Finally, it is essential to set the same tile size ```tile_size``` as used during training.


## Pretrained Models

We provide the checkpoints of two ResDepth-stereo models used to test geographical generalization between Berlin and 
Zurich (see Section *Geographical Generalization Across Cities* in the paper). Furthermore, we provide the checkpoint 
of our ResDepth-stereo multi-city model (see Section *Multi-city Model* in the paper).

To download these models, please run:
```shell
(resdepth) $ bash ./scripts/download_pretrained_models.sh
```

Additionally, we provide all the models used in the ablation studies (see Section *Influence of Image Guidance* in 
the paper).

To download these models, please run:
```shell
(resdepth) $ bash ./scripts/download_pretrained_models_ablations.sh
```

All the models will be downloaded and extracted to ```./logs/pretrained_models/``` and ```./logs/pretrained_models_ablations/```.


## Demo

Due to the commercial nature of VHR imagery, we cannot share our complete datasets. In this demo, we thus provide two 
variants of DSM refinement using a single DSM patch of 256&times;256 pixels only (64&times;64 m in world coordinates).

### Example Dataset

To download the demo, please run:
```shell
(resdepth) $ bash ./scripts/download_demo.sh
```

The data, configuration files, and pretrained models will be downloaded and extracted to ```./demo/```.

The data is stored in ```./demo/data/```:
```shell
./demo/data/
├── dsm
│   ├── DSM_Zurich_ground_truth.tif
│   └── DSM_Zurich_initial.tif
├── image_selection
│   ├── imagelist.txt
│   ├── pairlist_simple.txt
│   └── pairlist_generalized.txt
├── mask
│   └── Zurich_building_mask.tif
└── satellite_views
    ├── 15MAR17102414-P1BS-502980289080_01_P002.tif
    ├── 17OCT07103158-P1BS-501653123070_01_P002.tif
    ├── 18APR10105411-P1BS-502091706050_01_P002.tif
    ├── 18JAN29104120-P1BS-502980288020_01_P005.tif
    └── 18MAR24105605-P1BS-501687882040_02_P006.tif
```

We provide an initial DSM patch ```DSM_Zurich_initial.tif```, the corresponding ground truth DSM patch 
```DSM_Zurich_ground_truth.tif```, and a building mask ```Zurich_building_mask.tif```. The example DSM is located 
in the test stripe of the region ZUR1 in Zurich. Furthermore, we provide five ortho-images stored in the subdirectory 
```./demo/data/satellite_views/```. The subdirectory ```./demo/data/image_selection/``` comprises the image list 
```imagelist.txt``` and two pair lists, where ```pairlist_simple.txt``` defines a single stereo pair and 
```pairlist_generalized.txt``` two stereo pairs.


### Pretrained Models

The pretrained model weights are stored in ```./demo/models/```:
```shell
./demo/models/
├── ResDepth-stereo
│   ├── checkpoints
│   │    └── Model_best.pth
│   ├── DSM_normalization_parameters.p
│   ├── Image_normalization_parameters.p
│   └── model_config.json
└── ResDepth-stereo_generalized
    ├── checkpoints
    │    └── Model_best.pth
    ├── DSM_normalization_parameters.p
    ├── Image_normalization_parameters.p
    └── model_config.json    
```

The subdirectory ```./demo/models/ResDepth-stereo/``` contains the pretrained weights of a ResDepth-stereo prior. 
We used the training stripes of the region ZUR1 in Zurich and the single stereo pair listed in 
```./demo/data/image_selection/pairlist_simple.txt``` for training.

Similarly, the subdirectory ```./demo/models/ResDepth-stereo_generalized/``` specifies the pretrained weights of a 
ResDepth-stereo prior that has learned to generalize to unseen viewing directions, lighting conditions, and urban styles. 
For training, we sampled training data from all training stripes in ZUR1, ZUR2, and ZUR3 and leveraged multiple stereo 
pairs that are different from the ones listed in ```./demo/data/image_selection/pairlist_generalized.txt```. 


### Running the Demo

Use the available test configuration ```./demo/configs/config_simple.json``` to refine the initial DSM using the simple 
ResDepth-stereo prior (tailored to the specific image characteristics and acquisition geometry of the training image pair):
```shell
(resdepth) $ python test.py ./demo/configs/config_simple.json
```

To refine the initial DSM using the generalized ResDepth-stereo prior, use the test configuration 
```./demo/configs/config_generalized.json``` and run:
```shell
(resdepth) $ python test.py ./demo/configs/config_generalized.json
```

### Results

The refined DSMs will be stored in ```./demo/results/ResDepth-stereo/``` and <nobr>```./demo/results/ResDepth-stereo_generalized/```</nobr>.
For visualization, you can use Python visualization packages or off-the-shelf DSM visualization software such as 
[Quick Terrain Reader](https://appliedimagery.com/download/) or [planlauf/TERRAIN](https://planlaufterrain.com/).

As reference, ```./demo/results_expected/``` stores the expected results.

Preview of the results:
![Demo](docs/DSMs_all.gif)

## Contact

If you run into any problems or have questions, please contact [Corinne Stucker](mailto:corinne.stucker@geod.baug.ethz.ch).


## Citations

If you find this code or work helpful, please cite:
```
@article{stucker2022resdepth,
title = {{ResDepth}: A deep residual prior for 3D reconstruction from high-resolution satellite images},
author = {Stucker, Corinne and Schindler, Konrad},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {183},
pages = {560--580},
year = {2022}
}
```


<a name="myfootnote1">[1]</a>: C. Stucker, K. Schindler, ResDepth: Learned residual stereo reconstruction, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020, pp. 707-716.
