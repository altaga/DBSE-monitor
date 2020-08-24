# DBSE-monitor

Drowsines, blind spot and attention monitor for driving or handling heavy machinery. Also detects objects at the blind spot via Computer Vision powered by Pytorch and the Jetson Nano. And has a crash detection feature.

<img src="https://i.ibb.co/4mx4LPK/Logo.png" width="1000">

# Table of contents

* [Introduction](#introduction)
* [Materials](#materials)
* [Connection Diagram](#connection-diagram)
* [Laptop Test](#laptop-test)
* [Jetson Setup](#jetson-setup)
* [Drowsiness Monitor](#drowsiness-monitor)
* [Blind Spot Monitor](#blind-spot-monitor)
* [The Final Product](#the-final-product)
* [Commentary](#commentary)
* [References](#references)

# Introduction:

We will be tackling the problem of drowsiness when handling or performing tasks such as driving or handling heavy machinery and the blind spot when driving.

The Center for Disease Control and Prevention (CDC) says that 35% of American drivers sleep less than the recommended minimum of seven hours a day. It mainly affects attention when performing any task and in the long term, it can affect health permanently.

<img src="https://www.personalcarephysicians.com/wp-content/uploads/2017/04/sleep-chart.png" width="1000">

According to a report by the WHO (World Health Organization) (2), falling asleep while driving is one of the leading causes of traffic accidents. Up to 24% of accidents are caused by falling asleep, and according to the DMV USA (Department of Motor Vehicles) (3) and NHTSA (National Highway traffic safety administration) (4), 20% of accidents are related to drowsiness, being at the same level as accidents due to alcohol consumption with sometimes even worse consequences than those.

<img src="https://media2.giphy.com/media/PtrhzZJhbEBm8/giphy.gif" width="1000">

A su vez la NHTSA menciona que el hecho de estar enojado puede generar una conduccion mas peligrosa y agresiva (5), poniendo en peligro la vida del conductor debido a estas alteraciones psicologicas.

<img src="https://i.ibb.co/YcWYJNw/tenor-1.gif" width="1000">

# Solution:

We will create a system that will be able to detect a person's drowsiness level, this with the aim of notifying the user about his state and if he is able to drive.

At the same time it will measure the driver’s attention or capacity to garner attention and if he is falling asleep while driving. If it positively detects that state (that he is getting drowsy), a powerful alarm will sound with the objective of waking the driver.

<img src="https://i.gifer.com/origin/7d/7d5a3e577a7f66433c1782075595f4df_w200.gif" width="1000">

Additionally it will detect small vehicles and motorcycles in the automobile’s blind spots.

<img src="https://thumbsnap.com/s/Wy5w7JPR.jpg?1205" width="600">

In turn, the system will have an accelerometer to generate a call to the emergency services if the car had an accident to be able to attend the emergency quickly.

Debido a que un estado psicologico alterado podria generar una posible conduccion peligrosa, cuidamos el estado de el conductor mediante el analisis de emociones de su rostro y utilizando musica que al conductor le pueda genera una respuesta positiva.

<img src="https://i.ibb.co/xX4G7Yd/dondraper-car.gif" width="1000">

Current Solutions:

- Mercedes-Benz Attention Assist uses the car's engine control unit to monitor changes in steering and other driving habits and alerts the driver accordingly.

- Lexus placed a camera in the dashboard that tracks the driver's face, rather than the vehicle's behavior, and alerts the driver if his or her movements seem to indicate sleep.

- Volvo's Driver Alert Control is a lane-departure system that monitors and corrects the vehicle's position on the road, then alerts the driver if it detects any drifting between lanes.

- Saab uses two cameras in the cockpit to monitor the driver's eye movement and alerts the driver with a text message in the dash, followed by a stern audio message if he or she still seems sleepy.

As you can see these are all premium brands and there is not a single plug and play system that can work for every car. This, is our opportunity as most cars in the road are not on that price range and do not have these systems.

# Materials:

Hardware:
- NVIDIA Jetson Nano.                                x1.
https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-nano/
- Power Inverter for car.
https://www.amazon.com/s?k=power+inverter+truper&ref=nb_sb_noss_2
- ESP32.
https://www.adafruit.com/product/3405
- OLED display.
https://www.amazon.com/dp/B072Q2X2LL/ref=cm_sw_em_r_mt_dp_U_TMGqEb9YAGJ5Q
- Any Bluetooth Speaker or Bluetooth Audio Car System. x1.
https://www.amazon.com/s?k=speaker&s=price-asc-rank&page=2&qid=1581202023&ref=sr_pg_2
- USB TP-Link USB Wifi Adapter TL-WN725N.            x1.
https://www.amazon.com/dp/B008IFXQFU/ref=cm_sw_em_r_mt_dp_U_jNukEbCWXT0E4
- UGREEN USB Bluetooth 4.0 Adapter                   x1.
https://www.amazon.com/dp/B01LX6HISL/ref=cm_sw_em_r_mt_dp_U_iK-BEbFBQ76BW
- HD webcam                      .                   x1.
https://canyon.eu/product/cne-cwc2/
- 32 GB MicroSD Card.                                x1.
https://www.amazon.com/dp/B06XWN9Q99/ref=cm_sw_em_r_mt_dp_U_XTllEbK0VKMAZ
- 5V-4A AC/DC Adapter Power Supply Jack Connector.   x1.
https://www.amazon.com/dp/B0194B80NY/ref=cm_sw_em_r_mt_dp_U_ISukEbJN7ABK3
- VMA204.                                            x1.
https://www.velleman.eu/products/view?id=435512

Software:
- Pytorch:
https://pytorch.org/
- JetPack 4.3:
https://developer.nvidia.com/jetson-nano-sd-card-image-r3231
- YOLOv3:
https://pjreddie.com/darknet/yolo/
- OpenCV:
https://opencv.org/
- Twilio:
https://www.twilio.com/
- Arduino IDE:
https://www.arduino.cc/en/Main/Software
- Mosquitto MQTT:
https://mosquitto.org/

# Connection Diagram:

This is the connection diagram of the system:

<img src="https://i.ibb.co/Bqq3p6b/Esquema.png" width="1000">

# Laptop Test:

To test the code on a computer, the first step will be to have a python environments manager, such as Python Anaconda.

https://www.anaconda.com/distribution/

## Environment Creation:

### Pytorch

First we will create a suitable enviroment for pytorch.

    conda create --name pytorch

To activate the enviroment run the following command:

    activate pytorch

In the case of Anaconda the PyTorch page has a small widget that allows you to customize the PyTorch installation code according to the operating system and the python environment manager, in my case the configuration is as follows.

https://pytorch.org/

<img src="https://i.ibb.co/6RMJp5F/image.png" width="800">

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
    
### Dependencies
    
The other packages we need are the following:

    pip install opencv-python matplotlib tqdm python-vlc Pillow
    
Anyway we attach the file requirements.txt where all packages come in our environment.

### Jupyter Notebook

Inside the **Drowsiness**, **Emotion detection** and **YoloV3** folders, you will find a file "Notebook.ipynb" which contains the code to run the programs in jupyter notebook, however I attach in each folder a file called "notebook.py" with the code in format **. py **.

    conda install -c conda-forge notebook

Command to start jupyter notebook

    jupyter notebook

# Summary and mini demos:

All the demos that we are going to show are executed from a jupyter notebook and are focused on showing the functionality of the AI models, the demo with the hardware is shown at the end of the repository. [Demo](#epic-demo)

## Drowsiness Monitor:

La funcion de esta modelo es realizar una deteccion de distraccion o cerrado de ojos del conductor por mas de 2 segundos o esta distraido del camino (ejemplo, mirando el celular).

<img src="https://i.ibb.co/sQVStkj/Esquema-3.png" width="1000">

Code: https://github.com/altaga/DBSE-monitor/blob/master/Drowsiness/Notebook.ipynb

Video: Click on the image
[![Torch](https://i.ibb.co/4mx4LPK/Logo.png)](https://youtu.be/9Degq6HjrGE)

## Driving Monitor:

La funcion de esta modelo es realizar una deteccion objetos que esten a menos de 3 metros del auto en el punto ciego.

<img src="https://i.ibb.co/Xpd9rs8/Esquema-2.png" width="1000">

Code: https://github.com/altaga/DBSE-monitor/blob/master/YoloV3/Notebook.ipynb

Video: Click on the image
[![Torch](https://i.ibb.co/4mx4LPK/Logo.png)](https://youtu.be/auCgnU7oglc)

## Emotion Monitor:

La funcion de esta modelo es detectar las emociones del conductor en todo momento y mediante respuestas musicales (canciones) tratar de corregir el estado mental de el conductor con el fin de mantenerlo neutral o feliz.

<img src="https://i.ibb.co/dkfMKh7/Esquema-5.png" width="1000">

Code: https://github.com/altaga/DBSE-monitor/blob/master/Emotion%20detection/Notebook.ipynb

Video: Click on the image
[![Torch](https://i.ibb.co/4mx4LPK/Logo.png)](https://youtu.be/auCgnU7oglc)

# Jetson Nano Setup:





# Drowsiness Monitor:

Let's go through a revision of the algorithms and procedures of both CV systems (Drowsiness and alert on one side and Blind spot detection on the other). The installation is remarkably easy as I have already provided an image for the project.

ALL the code is well explained in the respective Github file: https://github.com/altaga/Torch-Drowsiness-Monitor

Look for the two Python files: Drowsiness and Yolo:

https://github.com/altaga/Torch-Drowsiness-Monitor/blob/master/Drowsiness/check.py

https://github.com/altaga/Torch-Drowsiness-Monitor/blob/master/YoloV3/detect.py

Those are the two that make all the magic happen.

Please take a look at it for extensive explanation and documentation.

The sleep monitor uses the following libraries:

- OpenCV:
- Image processing. 
    - (OpenCV) Haarcascades implementation. 
    - (OpenCV) Blink eye speed detection.
    - (Pytorch) Eye Status (Open / Close)
- Pygame: 
    - Player sound alert.
- Smbus:
    - Accelerometer reading.
- Twilio:
    - Emergency notification delivery.
- Requests:
    - Geolocation

The flicker detection algorithm is as follows:

- Detection that there is a face of a person behind the wheel:

<img src="https://i.ibb.co/ZMvwvfp/Face.png" width="600">

- If a face is detected, perform eye search.

<img src="https://i.ibb.co/StK0t2x/Abiertos.png" width="600">

- Once we have detected the eyes, we cut them out of the image so that we can use them as input for our convolutional PyTorch network.

<img src="https://i.ibb.co/0FYT0DN/Abiertoss.png" width="600">

- The model is designed to detect the state of the eyes, therefore it is necessary that at least one of the eyes is detected as open so that the algorithm does not start generating alerts, if it detects that both eyes are closed for at least 2 seconds , the alert will be activated, since the security of the system is the main thing, the algorithm has a second layer of security explained below.

- Because a blink lasts approximately 350 milliseconds then a single blink will not cause problems, however once the person keeps blinking for more than 2 or 3 seconds (according to our criteria) it will mean for the system that the person is falling asleep. Not separating the eyes from the road being one of the most important rules of driving.

<img src="https://i.ibb.co/kQ12W79/alert1.png" width="600">
<img src="https://i.ibb.co/LdVD7v2/alert2.png" width="600">

- Also during the development I found a incredible use case, when one turns to look at his cell phone, the system also detects that you are not seeing the road. This being a new aspect that we will be exploring in future versions of the system to improve detection when a driver is not looking at the road and is distracted. But, for now it is one of those unintended great finds.

<img src="https://i.ibb.co/mHZ4VdX/Cel.png" width="600">
<img src="https://i.ibb.co/3k512YS/cel2.png" width="600">

Whether it's because of the pytorch convolutional network model or the Haarcascades, the monitor will not allow you to take your eyes off the road, as it is extremely dangerous to do that while driving.

<img src="https://i.ibb.co/D84YbYb/Whats-App-Image-2020-03-16-at-12-35-40.jpg" width="600">

# Blind Spot Monitor:

The blind Spot monitor uses the following libraries:

- OpenCV:
    - Image processing
    - CDNN implementation.
        - The weights of the CDNN were obtained from YoloV3 and imported by PyTorch.
- MQTT with Mosquitto: 
    - Communication with the ESP32.

In this algorithm we use the detection of objects using PyTorch, YoloV3 and OpenCV which allows the use of CDNN.

<img src="https://i.ibb.co/S3Wz9Sc/process2.jpg" width="600">

In this case, I created an algorithm to calculate the approximate distance at which an object is located using YOLOV3, the algorithm is as follows.

We calculate the aperture and distance of our camera, in my case it is 19 centimeters in aperture at a distance of 16 cm.

It means that an object 19 cm wide can be at least 16 cm away from the camera so that it can fully capture it, in this case the calculation of the distance of any object becomes a problem of similar triangles.

<img src="https://i.ibb.co/cC8wbb6/esquemadist.png" width="1000">

In this case, since our model creates boxes where it encloses our detected objects, we will use that data to obtain the apparent width of the object, in this case the approximate width will be stored in a variable during the calculation.

<img src="https://i.ibb.co/jRf5xV3/exmaple.png" width="1000">

If we combine both terms we get the following equation.

<img src="https://i.ibb.co/8c8dBjB/Code-Cogs-Eqn-1.gif" width="600">

In this case for the objects that we are going to detect, we approximate the following widths in centimeters.

    {
        person:70,
        car:220,
        motorcycle:120,
        dogs:50 
    }

Here is an example of its operation with the same image.

<img src="https://i.ibb.co/wdf45wq/prockess2.jpg" width="1000">

Testing the algorithm with the camera.

<img src="https://i.ibb.co/d2dzt60/process.jpg" width="1000">
<img src="https://i.ibb.co/qjTc7rX/process1.jpg" width="1000">

Once having this distance, we will filter all distances that are greater than 2 meters, this being a safe distance to the car at the blind spot.

<img src="https://i.ibb.co/mzNdqVp/Whats-App-Image-2020-03-16-at-13-57-36.jpg" width="600">

In turn, we will determine in which of the 2 blind spots the object is, right or left. Depending on the object and the side, the information will be sent via Mosquitto MQTT to the display. For example a car on the left side:

<img src="https://i.ibb.co/dMcL9gn/20200210-203839.jpg" width="600">


# The Final Product:

Product:

<img src="https://i.ibb.co/gJB4f6R/20200210-212714.jpg" width="800">
<img src="https://i.ibb.co/99tCmt8/Whats-App-Image-2020-03-16-at-12-22-56.jpg" width="800">
<img src="https://i.ibb.co/sKLmfKq/Whats-App-Image-2020-03-16-at-12-22-57.jpg" width="800">

Product installed inside the car:

<img src="https://i.ibb.co/yQgJGfk/Whats-App-Image-2020-03-16-at-14-03-07-1.jpg" width="800">
<img src="https://i.ibb.co/6J5jSB5/Whats-App-Image-2020-03-16-at-14-03-07.jpg" width="800"> 

Notifications:

<img src="https://i.ibb.co/VNWzJ37/Screenshot-20200210-212306-Messages.jpg" width="600">

### Epic DEMO:

Video: Click on the image
[![Car](https://i.ibb.co/4mx4LPK/Logo.png)](https://youtu.be/X51jBfcTQxg)

Sorry github does not allow embed videos.

# Commentary:

I would consider the product finished as we only need a little of additional touches in the industrial engineering side of things for it to be a commercial product. Well and also a bit on the Electrical engineering perhaps to use only the components we need. That being said this functions as an upgrade from a project that a couple friends and myself are developing and It was ideal for me to use as a springboard and develop the idea much more. This one has the potential of becoming a commercially available option regarding Smart cities as the transition to autonomous or even smart vehicles will take a while in most cities.

That middle ground between the Analog, primarily mechanical-based private transports to a more "Smart" vehicle is a huge opportunity as the transition will take several years and most people are not able to afford it. Thank you for reading.

## References:

Links:

(1) https://medlineplus.gov/healthysleep.html

(2) http://www.euro.who.int/__data/assets/pdf_file/0008/114101/E84683.pdf

(3) https://dmv.ny.gov/press-release/press-release-03-09-2018

(4) https://www.nhtsa.gov/risky-driving/drowsy-driving

(5) https://www.nhtsa.gov/risky-driving/speeding
