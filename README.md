# MorphoGAN
Image style transfer tool that transforms your content images into artwork inspired by famous artistic styles. Built using Generative Adversarial Networks (GANs) and Adaptive Instance Normalization (AdaIN), this project allows you to merge your images with the artistic vision of iconic works from renowned artists.

## Features
Style Transfer: Apply different art styles to your content images.
Customizable Parameters: Adjust the content weight, style weight, and epochs for the best results.
Easy-to-Use Interface: A simple web-based interface for uploading content and style images.
Optimized Performance: Efficient training and fine-tuning using pretrained models.

## How I Came Up with AdaIN for Style Transfer
The key idea behind AdaIN is to align the statistical properties (mean and standard deviation) of the content and style features while maintaining the structure of the content. By normalizing the content features and then adapting them to the style's mean and standard deviation, AdaIN ensures that the content image retains its essence, while the style image contributes its artistic features without distortion.
To implement this in MorphoGAN, I combined the AdaIN technique with a pre-trained VGG19 network, which is widely used for feature extraction in style transfer. 

## Docker Setup
Build Docker Image: docker build -t morphogan .
Run Docker Container:docker run -p 5000:5000 morphogan
The application will be available at http://127.0.0.1:5000.

## Parameters:
content_weight: The importance of the content in the generated image. Default is 1.0.<br>
style_weight: The importance of the style. Default is 1000.
epochs: Number of epochs to train the model. Default is 100.
