# EEC174BY - Machine Learning Senior Design Project 
- Team Biometric Car
- Team Members: Andrew Kim, Jiakang Pei, Logain Abdelhafiz

## Setup on Computer
1. Clone this repository
2. Install required dependencies using

```pip3 install -r requirements.txt```   *There may be additional dependencies required. Please use pip install accordingly.

3. Run ```python3 main.py``` in the root directory

## Setup on Raspberry Pi

<img src="https://github.com/kim15096/eec174-biometric-car/assets/14260072/83e1c624-2833-45c5-a15c-a402322c5977" width="300" height="300">

Requirements:
- Raspberry Pi 4 (other models not tested)
- USB Webcam (at least 30fps)
- HDMI Cable / Ethernet Cable (if using VNC)
- Power cable (comes with Raspberry Pi)
- Cooling Unit for CPU
- Micro-sd card (at least 32GB)
- Keyboard + Mouse (USB A or Bluetooth)
- USB Speaker

Setting up:
- Flash Raspbian OS (64-bit): https://www.raspberrypi.com/news/raspberry-pi-imager-imaging-utility/
- Insert micro-SD card into Raspberry Pi
- Connect power cable and HDMI cable to external monitor
- Connect to the internet and Bluetooth peripherals (mouse and keyboard)
- Connect Webcam and install the cooler unit
- Clone github repository and install required libraries
- Run ```python3 main.py``` in the root directory

*Overclocking the Raspberry Pi (Strongly Recommended):
- Refer to this article: https://beebom.com/how-overclock-raspberry-pi-4/
- Overclock it to 2GHz (arm_freq = 2000, gpu_freq = 750, over_voltage = 6)

SSH into Pi:
- Connect ethernet cable from computer to Pi
- Download VNC on computer
- Find ip address of ethernet connection using ```ifconfig```
- Run ```ssh hostname@ip_address``` on computer's terminal (MacOS and Linux only)
- Connect using the VNC client and you will be granted access to view Pi's display
  
## Referenced Work
- Face Detection: https://github.com/e-candeloro/Driver-State-Detection.git (MIT License) Based on the paper: https://www.researchgate.net/publication/327942674_Vision-Based_Driver's_Attention_Monitoring_System_for_Smart_Vehicles
- Hand Detection: 
