# Laptop Test:

To test the code on a computer, the first step will be to have a python environments manager, such as Python Anaconda.

https://www.anaconda.com/distribution/

## Environment Creation:

First we will create a suitable enviroment for pytorch.

    conda create --name pytorch

To activate the enviroment run the following command:

    activate pytorch

In the case of Anaconda the PyTorch page has a small widget that allows you to customize the PyTorch installation code according to the operating system and the python environment manager, in my case the configuration is as follows.

https://pytorch.org/

<img src="https://i.ibb.co/6RMJp5F/image.png" width="800">

    conda install pytorch torchvision spyder cudatoolkit=10.1 -c pytorch

## Support Libraries:

In addition to the aforementioned command, it is necessary to install the following libraries so that the codes can be executed without problem.

- OpenCV
- pillow
- requests
- twilio
- pygame
- matplotlib
- paho-mqtt

Code to install all the libraries:

    pip install opencv-python Pillow requests twilio pygame matplotlib paho-mqtt

## Model Creation:

Inside the "https://github.com/altaga/Torch-Drowsiness-Monitor/tree/master/Drowsiness/Model" folder our model called "BlinkModel.t7" already exists, which is the one I use for all tests, however the model can be trained by yourself with the code called "train.py" in the folder "https://github.com/altaga/Torch-Drowsiness-Monitor/tree/master/Drowsiness".

The database that was used, is a database with 4846 images of left and right eyes, open and closed, where approximately half are open and closed so that the network was able to identify the state of the eyes, the database is in the following folder:

https://github.com/altaga/Torch-Drowsiness-Monitor/tree/master/Drowsiness/dataset/dataset_B_Eye_Images

The training has the following parameters as input.

- input image shape: (24, 24)
- validation_ratio: 0.1
- batch_size: 64
- epochs: 40
- learning rate: 0.001
- loss function: cross entropy loss
- optimizer: Adam

In the first part of the code you can modify the parameters, according to your training criteria.

Drowsiness Monitor:

- https://github.com/altaga/Torch-Drowsiness-Monitor/blob/master/Drowsiness/train.py

Video: Click on the image
[![Torch](https://i.ibb.co/1MC19TG/Logo.png)](https://youtu.be/y87Hht7-fkE)

## File Changes:

The codes to run on the laptop have the following modifications:

- The code segments that involve reading the accelerometer sensor are commented.
- The MQTT section is commented, instead the MQTT messages are displayed in the python console.
- The crash notification section is commented because it depends on the Twilio configuration and the use of the accelerometer.

The codes that are executed in the computer are the following and a video of how they are executed in real time with Anaconda Spyder (package that we previously installed):

To open the Spyder IDE write in the anaconda command console:
    spyder

Drowsiness Monitor:

- https://github.com/altaga/Torch-Drowsiness-Monitor/blob/master/Drowsiness/computer.py

Video: Click on the image
[![Torch](https://i.ibb.co/1MC19TG/Logo.png)](https://youtu.be/9Degq6HjrGE)

YoloV3:

- https://github.com/altaga/Torch-Drowsiness-Monitor/blob/master/YoloV3/computer.py

Video: Click on the image
[![Torch](https://i.ibb.co/1MC19TG/Logo.png)](https://youtu.be/auCgnU7oglc)

Since we could check that all the codes work, we can go to the configuration of our hardware to make our product.
