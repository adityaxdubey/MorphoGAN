# MorphoGAN
Image style transfer tool that transforms your content images into artwork inspired by famous artistic styles. Built using Generative Adversarial Networks (GANs) and Adaptive Instance Normalization (AdaIN), this project allows you to merge your images with the artistic vision of iconic works from renowned artists.

##Features
Style Transfer: Apply different art styles to your content images.
Customizable Parameters: Adjust the content weight, style weight, and epochs for the best results.
Easy-to-Use Interface: A simple web-based interface for uploading content and style images.
Optimized Performance: Efficient training and fine-tuning using pretrained models.

##Docker Setup
Build Docker Image: docker build -t morphogan .
Run Docker Container:docker run -p 5000:5000 morphogan
The application will be available at http://127.0.0.1:5000.

##Parameters:
content_weight: The importance of the content in the generated image. Default is 1.0.
style_weight: The importance of the style. Default is 1000.
epochs: Number of epochs to train the model. Default is 100.
