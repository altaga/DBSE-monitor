import cv2
import torch.hub
import os
import model
from PIL import Image
from torchvision import transforms
from grad_cam import BackPropagation
import pygame
import time 
#import smbus
#import requests
#from twilio.rest import Client
#import urllib.request
#import json

# Alarm sound file
file = 'alarm.mp3'
# Sound player start
pygame.init()
pygame.mixer.init()

"""
class MMA7455():
    bus = smbus.SMBus(1)
    def __init__(self):
        self.bus.write_byte_data(0x1D, 0x16, 0x55) # Setup the Mode
        self.bus.write_byte_data(0x1D, 0x10, 0) # Calibrate
        self.bus.write_byte_data(0x1D, 0x11, 0) # Calibrate
        self.bus.write_byte_data(0x1D, 0x12, 0) # Calibrate
        self.bus.write_byte_data(0x1D, 0x13, 0) # Calibrate
        self.bus.write_byte_data(0x1D, 0x14, 0) # Calibrate
        self.bus.write_byte_data(0x1D, 0x15, 0) # Calibrate
    def getValueX(self):
        return self.bus.read_byte_data(0x1D, 0x06)
    def getValueY(self):
        return self.bus.read_byte_data(0x1D, 0x07)
    def getValueZ(self):
        return self.bus.read_byte_data(0x1D, 0x08)


# Crash Sensibility
sens=30

# Sending SMS if Crash Detected


def send():
    # Your Account SID from twilio.com/console
    account_sid = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    # Your Auth Token from twilio.com/console
    auth_token  = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    client = Client(account_sid, auth_token)
    phone = "+XXXXXXXXXXXX"
    print('crash')
    send_url = 'http://ip-api.com/json'
    r = requests.get(send_url)
    j = json.loads(r.text)
    text="The Driver Crash Here: "
    text+="http://maps.google.com/maps?q=loc:{},{}".format(j['lat'],j['lon'])
    print(text)
    message = client.messages.create(to=phone, from_="++XXXXXXXXXXXX",body=text)
    print(message.sid)
    time.sleep(10)
    stop()


# Accelerometer Declaration
mma = MMA7455()

# Obtaining the X, Y and Z values.

xmem=mma.getValueX()
ymem=mma.getValueY()
zmem=mma.getValueZ()
x = mma.getValueX()
y = mma.getValueY()
z = mma.getValueZ()


# Creating the base accelerometer values.

if(xmem > 127):
    xmem=xmem-255
if(ymem > 127):
    ymem=ymem-255
if(zmem > 127):
    zmem=zmem-255
if(x > 127):
    x=x-255
if(y > 127):
    y=y-255
if(z > 127):
    z=z-255

"""

timebasedrow= time.time()
timebasedis= time.time()
timerundrow= time.time()
timerundis= time.time()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
MyModel="BlinkModel.t7"

shape = (24,24)
classes = [
    'Close',
    'Open',
]

eyess=[]
cface=0

def preprocess(image_path):
    global cface
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])    
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        ...
    else:
        cface=1
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color = image[y:y+h, x:x+w]
        """
        Depending on the quality of your camera, this number can vary 
        between 10 and 40, since this is the "sensitivity" to detect the eyes.
        """
        sensi=20
        eyes = eye_cascade.detectMultiScale(face,1.3, sensi) 
        i=0
        for (ex,ey,ew,eh) in eyes:
            (x, y, w, h) = eyes[i]
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye = face[y:y + h, x:x + w]
            eye = cv2.resize(eye, shape)
            eyess.append([transform_test(Image.fromarray(eye).convert('L')), eye, cv2.resize(face, (48,48))])
            i=i+1
    cv2.imwrite('display.jpg',image) 
    

def eye_status(image, name, net):
    img = torch.stack([image[name]])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    actual_status = ids[:, 0]
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    #print(name,classes[actual_status.data], probs.data[:,0] * 100)
    return classes[actual_status.data]

def func(imag,modl):
    drow(images=[{'path': imag, 'eye': (0,0,0,0)}],model_name=modl)

def drow(images, model_name):
    global eyess
    global cface
    global timebasedrow
    global timebasedis
    global timerundrow
    global timerundis
    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join('Model', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    flag =1
    status=""
    for i, image in enumerate(images):
        if(flag):
            preprocess(image)
            flag=0
        if cface==0:
            image = cv2.imread("display.jpg")
            image = cv2.putText(image, 'No face Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite('display.jpg',image)
            print("No face Detected")
            timebasedrow= time.time()
            timebasedis= time.time()
            timerundrow= time.time()
            timerundis= time.time()
        elif(len(eyess)!=0):
            eye, eye_raw , face = eyess[i]
            image['eye'] = eye
            image['raw'] = eye_raw
            image['face'] = face
            timebasedrow= time.time()
            timerundrow= time.time()
            for index, image in enumerate(images):
                status = eye_status(image, 'eye', net)
                if(status =="Close"):
                    print("Distracted")
                    timerundis= time.time()
                    if((timerundis-timebasedis)>1.5):
                        image = cv2.imread("display.jpg")
                        image = cv2.putText(image, 'Distracted', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.imwrite('display.jpg',image)
                        pygame.mixer.music.load(file)
                        pygame.mixer.music.play()
                
                else:
                    pygame.mixer.music.stop()
                    timebasedis= time.time()          
        else:
            timerundrow= time.time()
            if((timerundrow-timebasedrow)>3):
                pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                image = cv2.imread("display.jpg")
                image = cv2.putText(image, 'Drowsy', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imwrite('display.jpg',image)
                print("Drowsy")

def main():
    global eyess
    global cface    
    eyess=[]
    cface=0
    ret, img = cap.read() 
    cv2.imwrite('img.jpg',img) 
    func('img.jpg',MyModel)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    timebasedrow= time.time()
    timebasedis= time.time()
    timerundrow= time.time()
    timerundis= time.time()
    while 1:
        """
        x = mma.getValueX()
        y = mma.getValueY()
        z = mma.getValueZ()
        if(x > 127):
            x=x-255
        if(y > 127):
            y=y-255
        if(z > 127):
            z=z-255
        # Send sms if crash
        if(abs(xmem-x)>sens or abs(ymem-y)>sens or abs(zmem-z)>sens):
            send()
            print('Crash')
        """
        
        main()
        img = cv2.imread("display.jpg")
        cv2.imshow('image',img) 
        k = cv2.waitKey(30) & 0xff
        

