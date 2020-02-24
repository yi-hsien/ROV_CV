import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# perform colour dominant tests
# 2 long colours
# 6 short colours
# image with 4 colours is on top
# second image with 1 long colour is bottom,
# bottom box's long colour touches the matching side of the top box

# get two small boxes, match to correct side of bottom box
# align remaining long box to the right of the right small box


"""
    convert each of the five faces of the box into arrays 
"""


folders = glob.glob('*')
print(folders)
imagenames_list = []
for f in folders:
    if '.png' in f:
        imagenames_list.append(f)

read_images = []
print(imagenames_list)

for image in imagenames_list:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    read_images.append(img)

"""
<<<<<<< HEAD
  
"""
=======
    do colour shit or some shit
    then match and parse images to one big image
    print image
    
    
    1 arduino mega mini for:
     
     ontrolling the six thrusters according to the data received from the
    joystick.
    •Receiving the PH, pressure, and temperature sensors data. 
    Then, 
    processing the data and sending it to the Co-Pilot on the Control Unitlaptop.
    
    •Sending the signal for triggering the solenoid responsible for the pneumatic 
    arm function.
    
    •Sending the Pulse Width Modulation (PWM)signal and the directional 
    signal for the two Cryton 10A 5-30v dual channel DC motor drivers that drive 4 DC 
    motors. 
    
    These 4 dc motors are divided according to their functions as follows:•One 
    is used for the main ROV DC arm.
    
    •Other two are used for the micro ROV and its tether 
    roller.•The last one serves as turning gear for the pneumatic arm
     
     
     
     
     the motors
    LED lights
    claw
    1 camera for Front -> to computer, do CV stuff and spit back shit
    1 camera for Down -> to computer, do CV stuff and spit back shit
    joystick
    
    
    1 camera for microROV?
    
"""
>>>>>>> 99522edad66f9e2aa5638071901099159090f730
