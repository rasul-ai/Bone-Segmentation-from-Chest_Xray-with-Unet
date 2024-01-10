# Bone-Segmentation-with-Unet
**Author:** [Md Rasul Islam Bapary]  
**Date:** [27.12.2023]

In this repository I have tried to implement a segmentation model with Unet of my own. I have inspired to implement this from a paper ***MDU-Net: A Convolutional Network for Clavicle and Rib Segmentation from a Chest Radiograph*** [Link](https://www.hindawi.com/journals/jhe/2020/2785464/). My code architecture is different from the architecture of the original paper. Basically, I have done this work to increase my skill in the field of deep learning.

## I have created a slide on this project. Please view the slide to know more.
[Slide Link](https://docs.google.com/presentation/d/1xe10kFKCvv7x1hZEjlrAgQQvUxf1pVj4xnaASz22-7U/edit?usp=sharing)

## Original Image
I am uploading two samples of my dataset that I have used for my experiment.
![image_1](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/person10.jpg)
![image_2](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/person13.jpg)

## Annotation samples
To annotate images I used Labelme tool. Here is the interface of the Labelme Interface.
![Labelme_Interface](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-22-44.png)

The annotation files look like this,
![Annotation_s1](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-22-54.png)
![Annotation_s2](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-23-03.png)

## Evaluation with validation and test set
I have also evaluated my model with the validation and test set. And the result shows very good result. Here is some evaluation result.
![Val_1](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-23-24.png)
![Test_1](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-23-31.png)
![Test_2](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-23-35.png)

## Evaluation with benchmark dataset
The benchmark dataset is collected from internet. Let me clear one thing, these set contains images that is completely different from the train/test/validation set. Here is also the result is quite good except that there are some noise in the mask image after inferencing. Due to the reason of using only 13 image to train the model, I think the result is little poor. More annotation will help to overcome the problem and increase performance of the model.
![Benchmark_1](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-23-43.png)
![Benchmark_2](https://github.com/rasul-ai/Bone-Segmentation-with-Unet/blob/main/images/Screenshot%20from%202024-01-10%2022-23-49.png)

### I will upload the original preprocessed dataset soon.
