#!/usr/bin/python
'''
author: Tobias Weis, 2014, [weis DOT tobi AT googlemail.com]

This module communicates with a SICK PLS 101-312 Laserscanner.
Although the protocol is pretty similar to the LM-200, 
I do not know if this would work.

Usage example:

sick = SICK("/dev/ttyUSB0")
while True:
    if sick.get_frame():
        print sick.cartesian
        cv2.imshow("img", sick.image)
        cv2.waitKey(5)


Calling sick.get_frame will automatically fill sick.cartesian and sick.image:

sick.cartesian contains coordinates in the following format:
    [x-coord (in cm), y-coord (in cm), distance (in cm)]

sick.image is an opencv-color-image showing the distances from the scanner

Stuff that helped me along the way (w/ protocols etc.):
- http://www.cs.cmu.edu/afs/cs/project/lri/Meteorite99/Software/src/hardwareDrivers/sick/sickDrv.c
- http://www.cs.cmu.edu/afs/cs/project/lri/Meteorite99/Software/src/hardwareDrivers/sick/sickDefs.h
- https://github.com/hawesie/sicktoolbox/blob/master/c%2B%2B/drivers/pls/sickpls/SickPLS.hh
'''

import serial
import time
import sys
import math
from CRC16_SICK import *
import numpy as np

SICK_STX = 0x02
SICK_ACK = 0x06
SICK_NAK = 0x15
SICK_NACK = 0x92
SICK_STATUS = 0x31

#################################### GENERAL MESSAGE LAYOUT TO SICK:
SICK_START = 0x02 # start of transmission
SICK_DEST = 0x00 # dest adress
# payload-length is two addition bytes, makes total of 4 in the header
# now payload bytes
# checksum bytes (2)
###################################################################


class SICK():
    def __init__(self, port, debug = False, password="LASERFUN"):
        self._debug = debug
        self.password = password
        self.ser = serial.Serial(port, 9600)
        self.ser.baudrate = 9600
        self.ser.parity = serial.PARITY_ODD
        self.ser.stopbits = serial.STOPBITS_ONE
        self.ser.bytesize = serial.EIGHTBITS
        self.ser.xonxoff = True
        self.ser.rtscts = False
        self.ser.dsrdtr = False
        self.ser.timeout = .1
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.flush()
        self.crc_calc = CRC16_SICK()
        self.frame = None
        self.cartesian = None

        # I use this as start sequence to ensure everything is set
        if not self.test_status():
            print "Trying with baudrate 38400"
            self.ser.baudrate = 38400# checks for reply and changes baudrate if nec
            if not self.test_status():
                raise(Exception("Can not communicate with SICK. Please check connection, port and baudrate"))
        # self.reset() # reset and initilize scanner
        self.log_in() # login with password
        if self.ser.baudrate == 9600:
            self.set_38k() # set to 38k
            if not self.test_status():
                self.ser.baudrate = 9600
                self.set_38k()
                self.log_in()
        self.request_scan_mode() # request scanning
        time.sleep(1)
        self.set_op_mode() # set op-mode to enable scanning, wtf, dont know why

    def calc_distances(self):
        coords = np.empty((0,3))
        for i in range (0,361):
            sval = (self.frame[i*2+8]<<8 | self.frame[i*2+7]) & 0x1fff
            x =  sval * math.cos(float(i)/2.0*3.1415/180)
            y = sval * math.sin(float(i)/2.0*3.1415/180)
            coords = np.append(coords, np.array([[x,y,sval]]), axis=0)
        self.cartesian = coords

    def hexar2str(self, hexar):
        return ''.join([chr(c) for c in hexar])

    def parse_msg(self, msg):
        if self._debug:
            print [hex(x) for x in msg]
        if len(msg) > 0:
            response = 0
            if self._debug:
                print "Length: %d" % (len(msg))
            if msg[0] == SICK_STX:
                try:
                    response = msg[4]
                    if response == 0xb0:
                        if self.crc_calc.calcSICK(self.hexar2str(msg[:-2])) == msg[-2:]:
                            self.frame = msg[:-2]
                            self.calc_distances()
                        else:
                            print "CHECKSUM DOES NOT MATCH !"
                    elif response == 0xa0:
                        if self._debug:
                            print "Acknowledged switch operating mode"
                        response = 0xa0
                except:
                    #print "Error while parsing STX message"
                    pass
            elif msg[0] == SICK_NACK:
                print "NACK"
            elif msg[0] == SICK_NAK:
                print "Telegram invalid !"
            elif msg[0] == SICK_ACK:
                try:
                    msg = msg[1:]
                    response = msg[4]
                    data = msg[5:-2]
                    if response == 0xb2:
                        if data[0] == 0x02:
                            print "Warning: %s" % hex(data[1])
                        elif data[0] == 0x01:
                            print "Info: %s" % hex(data[1])
                        elif data[0] == 0x00:
                            print "Nothing Wrong"
                        elif data[0] == 0x03:
                            print "Error: %s" % hex(data[1])
                        elif data[0] == 0x04:
                            print "Fatal Error: %s" % hex(data[1])
                    elif response == SICK_NACK:
                        print "NACK Telegram !"
                    elif response == 0xa0:
                        if self._debug:
                            print "Response to switch operating mode"
                        if data[0] == 0x00:
                            if self._debug:
                                print "Successfull"
                        elif data[0] == 0x01:
                            print "Not possible, wrong password"
                        else:
                            print "Dont know, strange stuff: %s" % (hex(data[0]))
                except:
                    #print "Error while parsing ACK message"
                    pass

            if self._debug:
                print "."
            return response


    def create_and_send_msg(self, msg_to_send):
        self.ser.flush()
        msg = []
        msg.append(SICK_START)
        msg.append(SICK_DEST)
        length = len(msg_to_send)
        msg.append(length & 0x00ff) # split length to Hi and Lo byte
        msg.append(length >> 8)
        for i in msg_to_send:
            msg.append(i)
        crc = self.crc_calc.calcSICK(self.hexar2str(msg))
        msg.append(crc[0])
        msg.append(crc[1])
        if self._debug:
            print "Message: %s" % (''.join([hex(b)+" " for b in msg]))
        self.ser.write(self.hexar2str(msg))


    def request_scan_mode(self):
        print "----------- REQUESTING SCAN MODE:"
        while True:
            self.create_and_send_msg([0x36, 0x01, 10])

            asw = []
            while self.ser.inWaiting() > 0:
                asw.append(ord(self.ser.read(1)))
            if self.parse_msg(asw) == 0xa0:
                print "Scan request successfull"
                break
            else:
                time.sleep(.1)

    def set_op_mode(self):
        print "------------- SETTING OPERATION MODE"
        while True:
            self.create_and_send_msg([0x20, 0x24])
            asw = []
            while self.ser.inWaiting() > 0:
                asw.append(ord(self.ser.read(1)))
            if self.parse_msg(asw) == 0xa0:
                print "Operation mode changed"
                break
            else:
                time.sleep(.1)


    def test_status(self):
        count = 0
        while True:
            self.create_and_send_msg([SICK_STATUS])
            # ATTENTION !
            # IF WE DO NOT RECEIVE A STATUS MESSAGE,
            # SWITCH BAUDRATE TO HIGHER ONE
            # (FOR EXAMPLE WHEN WE ALREADY SET IT BEFORE,
            # BUT RESTARTED THE PROGRAM)
            count = count + 1
            if count == 10:
                print "No response, baudrate wrong ?!"
                return False
            asw = []
            while self.ser.inWaiting() > 0:
                asw.append(ord(self.ser.read(1)))
            if self.parse_msg(asw) == 0xb1:
                print "TEST: Status message successfull"
                break
            else:
                time.sleep(.5)
        return True


    def set_38k(self):
        print "Setting 38.4k baudrate"
        while True:
            self.create_and_send_msg([0x20, 0x40])
            asw = []
            while self.ser.inWaiting() > 0:
                asw.append(ord(self.ser.read(1)))
            if self.parse_msg(asw) == 0xa0:
                print "Speed set to 38k, changing speed of self.serial interface"
                self.ser.baudrate = 38400
                break
            else:
                time.sleep(.1)

    def log_in(self):
        print "Logging in"
        pwd = [ord(c) for c in self.password]
        while True:
            self.create_and_send_msg([0x20, 0x00]+ pwd)
            asw = []
            while self.ser.inWaiting() > 0:
                asw.append(ord(self.ser.read(1)))
            if self.parse_msg(asw) == 0xa0:
                print "logged in"
                break
            else:
                time.sleep(.1)

    def reset(self):
        print "Resetting SICK and initing"
        self.create_and_send_msg([10])
        print ("Waiting.")
        time.sleep(6)

    def get_frame(self):
        while True:
            asw = []
            start = False
            while len(asw) < (727 + 5):
                while self.ser.inWaiting() > 0:
                    r = ord(self.ser.read(1))
                    if (r == SICK_STX): # read until start-byte discovered
                        start = True
                    if start == True:
                        asw.append(r)
                time.sleep(.0001)
            if self.parse_msg(asw) == 0xb0:
                return True



