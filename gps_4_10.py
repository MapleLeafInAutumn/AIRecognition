import serial
import pynmea2
import time


ser  =serial.Serial("/dev/ttyUSB1", 9600, timeout=0.2)

while True:
    line = serial.readline()
    line = str(line, encoding='utf-8')
    if line.startswith('$GNGGA'):
        record = pynmea2.parse(line)
        print(record)