# Markerless Mice Tracking for Social Experiments

This is an implementation of the pipeline to track markerless mice which is described in the paper.

## Installation
The code has been run successfully on Windows with an NVIDIA GPU
1. Clone this repository
2. [Anaconda](https://www.anaconda.com/distribution/) is highly recommended to install Python 3
3. Install dependencies with our provided Anaconda environments
   ```bash
   conda env create -f conda-environments/environment_windowsGPU.yml
   ```
4. Activate the environment and run Jupyter notebook

   ```bash
   conda activate markerless_mice_tracking_windowsGPU
   jupyter notebook
   ```

## Apply the algorithm on your own data
To apply the algorithm to new videos which have significantly different settings with our settings described in the paper, we recommend you to
retrain Mask-RCNN and Deeplabcut models on your own data and update the model in the tracking pipeline. The workflow to train those models can be found in two Jupyter Notebooks as follows:
* *pipeline/deeplabcut_training.ipynb*
* *pipeline/mrcnn_training.ipynb*

Otherwise, the current pipeline in *pipeline/pipeline.ipynb* uses our pretrained Mask-RCNN and DeepLabCut models.

To track mice in new videos, you need to follow the steps in *pipeline/tracking_pipeline.ipynb*
* Input of the pipeline can be a video or a directory containing frames in sequence. 
* Output of the workflow are two csv files storing coordinates of snout and tailbase corresponding to two mice: *features_mouse1_ensemble.csv* and *features_mouse2_ensemble.csv*. Besides that,
all intermediate data generated are also saved.

## Pretrained models and video samples 
1. Mask-RCNN model
https://drive.google.com/uc?export=download&id=17jWmHP8lmNhjROprMN4Exd9lfl-svFCA
The model must be saved in the path:  *mrcnn_models/mask_rcnn_mouse_0025.h5*
2. Deeplabcut model
https://drive.google.com/uc?export=download&id=1HTDNhTKWRAxsFvK7m2dtNI5JgQAH_h2N
The zip file must be extracted and saved in the path:  *dlc_models/dlc_mice_tracking

3. Video examples and background photos:
https://drive.google.com/drive/folders/1W3NCg_woHhlSPrmJy37irR2qIrVqlJf9?usp=sharing

The current configuration allows you to track 2 mice in the videos, but you can expand to track more mice with proper configuration.

Also, the current pipeline in Jupyter Notebook is not implemented with multiprocessing. We will update this soon.