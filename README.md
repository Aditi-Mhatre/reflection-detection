# Detection of Light Reflections

This repository contains the code to detect light reflections for reactive planning in autonomous inspections. To solve the problem both image processing and deep learning processes are executed, with the focus on this project being on the semantic segmentation methods by using deep learning based methdologies. 

<img src="./images/more-results.jpg" alt="Results with Deep Learning Methods">

The objective of this thesis is to detect light reflections and aid in the reactive planning of autonomous inspections. Based on the semantic segmentation models (whose results are shown above), the coordinates of the reflections are calculated and returned in the post-processing. 

The results after inference are depicted on a sample image below:

<img src="./images/unet-inspect.png" alt="Inference">


## Run Code 

### Deep Learning

To run the code create a virtual enviroment and install the dependencies in requirements.txt:

```
python3 -m venv .myenv
source .myenv/bin/activate
pip3 install -r requirements.txt

```

To visualize the reflections and get the coordinates:
- Run [inference.py](./Deep-Learning/inference.py)


## Traditonal Methods

Traditional Methods are the basic image processing algorithms that are used for segmenting images. These algorithms can be thresholding, edge-based or region-based segmentation methods. 
The methods covered in this repository are:
- Thresholding algorithms: 
    - Global: segments based on a threshold value for the entire image
    - Local or Adaptive: applies different thresholding values to different parts of the image based on the local pixel values
- K-Means Clustering: identifies different clusters and classes based on how similar the data is (not tested)
- Canny Edge Detection: segments by detecting the edges (not tested)
- The code can be found in [Traditional Methods]()


## Deep Learning Methods

Deep learning methods refer to a subset of machine learning techniques that use neural networks with multiple layers to model and learn from large amounts of data. This method is useful for semantic segmentation where the image needs to be partioned and assigned to a class. In this project, the models generate a binary image, with reflection areas marked as white and non-reflected areas as black. 

There are five models implemented (across three datasets: Inspection, SHIQ, and WHU):
- U-Net
- AttentionUNet
- UNet++
- [UNETR](./Deep-Learning/unetr_new.py) 
- [UNETR-Attention Fusion (UNETR-AF)](./Deep-Learning/unetr_af.py)

Note:
-U-Net and AttentionUNet are saved in [model.py](./Deep-Learning/model.py)
-UNet++ is imported from the segmentation model library
-UNETR and UNETR-AF have their own files

All code for the models is found in:
[Deep Learning]()


### Inspection Dataset

A specialized dataset consisting of 1025 inspection images is created for this thesis. The images are captured under varying light conditions and a corresponding masks is generated to highlight the reflections. The dataset is split into Train (820 images) and Test (205 images). 

<img src="./extra-images/inspect-dataset.jpg" alt="Inspect Dataset">

***

