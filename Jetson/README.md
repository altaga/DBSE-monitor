# Jetson Nano Setup:

You must download the model from the link below and put it inside the YoloV3/weights folder for the code to work

https://pjreddie.com/media/files/yolov3.weights

## SD card Setup:

In this case we use an OS that already has the pytorch library and jupyter lab installed to facilitate the use of the jetson.

https://github.com/NVIDIA-AI-IOT/jetbot/wiki/software-setup

Format the SD card with SD Card Formatter and Flash the operating system in the SD with Etcher.

## Hardware Setup:

Because we will be using several hardware components such as cameras, an accelerometer, speakers and more, I highly recommend a power supply of at least 5V - 4A.

The only hardware device that is not plug and play is the accelerometer because this one uses a I2C protocol (TWI if you speak Atmel). That's why we have to check if the Jetson Nano has these pins available.

<img src="https://i.ibb.co/5x6bsWJ/bus.png" width="800">

In this case we will use the Bus 1 because of its proximity to the 5V and GND pins. This will allow us to create a soldered circuit like an Arduino shield, but this time for the Jetson Nano.

<img src="https://i.ibb.co/BP47wYY/Esquema-bb.png" width="800">

After soldering our circuit looks like so:


<img src="https://i.ibb.co/dbH7KDN/Whats-App-Image-2020-03-16-at-14-10-55-1.jpg" width="800">
<img src="https://i.ibb.co/F40q4Rh/Whats-App-Image-2020-03-16-at-14-10-55.jpg" width="800">

This next video show us how to setup our hardware:

Video: Click on the image
[![Setup](https://i.ibb.co/4mx4LPK/Logo.png)](https://youtu.be/I8z-k-uc0fk)

Curious Fact: The Nvidia Jetson Nano has the same IO pin distribution as a Raspberry Pi, so every shield for the Pi is backwards compatible with the Nano!

## WiFi Setup:

The device has to be first set up with a WiFi connection, because of the fact that there are a ton of libraries we have to install. During the boot portion of the OS installation, we will connect the Jetson to your private WiFi and next we will set up the WiFi from our mobile phone (For this project as we need a Cellular connection). The advantage of doing this is that the Nano will connect to the one available so we can easily control this from the outside. TLDR: Connect both your private WiFi and your Mobile phone during the first setup, you the system can recognize both automatically.

Boot WiFi:

<img src="https://i.ibb.co/mz9tbYq/Jetson-Wi-Fi-Boot.png" width="800">

Home WiFi IP:

IP: 192.168.0.23

<img src="https://i.ibb.co/dL0nTsf/image.png" width="800">

Mobile WiFi:

IP: 192.168.43.246

<img src="https://i.ibb.co/y5HZm42/Ip-Phone-New.png" width="800">

## Software Setup:

We need to access our jetson from any browser.

    jetson_ip:8888

The command that I use in MY case is:

    192.168.0.23:8888

Password:jetbot

<img src="https://i.ibb.co/93HBrXd/image.png" width="1000">

## Twilio Setup

### Create Account

Open a twilio account to be able to notify via SMS in case of a crash.

https://www.twilio.com/try-twilio

### Obtain your credentials

The credentials for the Python code appear immediately upon entering, save the SID and the token, we will configure them later.

<img src="https://i.ibb.co/4MjwrBH/twilio.png" width="1000">

### Verify your Personal Phone Number

When you signed up for your trial account, you verified your personal phone number. You can see your list of verified phone numbers on the Verified Caller IDs page.

Go to your Verified Caller IDs page in the console.

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/find_phone_nums_dash.width-500.png" width="300">

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/find_verified_callerIDs_link.width-500.png" width="1000">

Click on the red plus (+) icon to add a new number.

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/verified_caller_ids_page.width-800.png" width="1000">

Enter the phone number you wish to receive the notification from Twilio. 

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/verify_phone_num.width-500.png" width="1000">

**Note: You will need access to this device to receive the call or text with your verification code**

Enter the verification code. You’re now ready to text or call this number with your trial Twilio account.

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/phone_verification_code.width-500.png" width="1000">

### Get Twilio Phone Number

Go to Phone Numbers page in the console.

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/find_phone_nums_dash.width-500.png" width="300">

Go to Getting Started page in the console and click on "Get your first Twilio phone number".

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/get_your_trial_number.width-800.png" width="1000">

A number will be automatically offered, save this number for later and press "Choose this number" to finish setting up your number.

<img src="https://twilio-cms-prod.s3.amazonaws.com/images/Screen_Shot_2017-11-30_at_8.49.18_AM.width-800.png" width="1000">

## Mosquitto MQTT setup

Create your own Mosquitto MQTT.

https://mosquitto.org/

<img src="https://mosquitto.org/images/mosquitto-text-side-28.png" width="1000">

Una vez instalado el mqtt server este iniciara en el boot de la board.

## Support Libraries Setup:

- Pytorch (Preinstalled)

Los paquetes que vamos a necesitar para hacer funcionar el sistema seran los siguientes.

- OpenCV
- pillow
- torchvision
- requests
- twilio
- pygame
- matplotlib
- paho-mqtt

Code to install all the libraries:

    sudo apt-get install python-opencv
    sudo pip3 install opencv-python Pillow torchvision requests twilio matplotlib paho-mqtt smbus python-vlc

Password:jetbot

## Download the project:

To download the project in the jetson we must execute the following command

    git clone https://github.com/altaga/DBSE-monitor

# ESP32 Setup

Schematics of the OLED (128x32) and the ESP32.

<img src="https://i.ibb.co/n8G4Yk1/Oled.png" width="1000">

I grabbed a case and fit everything inside and this is how it looks:

<img src="https://i.ibb.co/WxqDyqB/1.jpg" width="1000">
<img src="https://i.ibb.co/qn3n70n/20200210-200045.jpg" width="1000">

These are the images it will show according to the CV detection, whether it sees a bike or a dog or anything else:

<img src="https://i.ibb.co/Htrf0Bm/20200210-203851.jpg" width="400"><img src="https://i.ibb.co/TRCLzjv/20200210-203909.jpg" width="400">
<img src="https://i.ibb.co/FmnvLtw/20200210-203826.jpg" width="400"><img src="https://i.ibb.co/dMcL9gn/20200210-203839.jpg" width="400">

If you want to add more images to the display, you have to use the following image2cpp converter, which will convert pixels into a code that Arduino IDE can use:

https://javl.github.io/image2cpp/

The web application gives us the code to copy and paste in our arduino project.

The device code is in the "Arduino" folder and all we have to do is configure the WiFi and MQTT credentials.

<img src="https://i.ibb.co/LJH9jhp/image.png" width="1000">