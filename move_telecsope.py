# Run this in a Jupyter notebook cell for debugging

import os
import win32com.client
import time
import numpy as np

# Main camera plate solve, binned 2x2:
# RA 12h 55m 33.6s,  Dec +03° 27' 42.6"
# Pos Angle +04° 34.7', FL 1178.9 mm, 1.59"/Pixel

main_plate = 1.59/2 # arcsec/pix
main_angle = 4.578333333333333 # CCW from N on east side of pier

# Guider (Binned 1x1)
# RA 07h 39m 08.9s,  Dec +34° 34' 59.0"
# Pos Angle +178° 09.5', FL 401.2 mm, 4.42"/Pixel

def test_scope():
    T = win32com.client.Dispatch("ASCOM.Simulator.Telescope")
    if T.Connected:
        print("	->Telescope was already connected")
    else:
        T.Connected = True
        if T.Connected:
            print("	Connected to telescope now")
        else:
            print("	Unable to connect to telescope, expect exception")

        T.Tracking = True
        T.SlewToCoordinates(12.34, 86.7)     # !!!pick coords currently in sky!
        print("RA = " , T.RightAscension)
        print("DE = " , T.Declination)
        T.Connected = False

# Pulse guide is an ASCOM camera method.  Not sure that is available
# to us with the camera in Maxim.  Ah, we have CCDCamera.GuiderMove.
# That does timed moves in particular direction.  It does not wait
# until return.

# It looks like someone has done some of this before.  Wonder how much
# of this I should use.
# http://wt5l.blogspot.com/2011/05/automated-astrophotography-with-python_28.html
# That basically looks like reprogramming stuff that MaxIM sets up,
# making it interactive from the command line on Python, which is
# totally not the direction I want to go.  But it is a good example
# and may have some style hints, such as name mangling for subclasses

# WOW!  This worked better than I thought!  It started MaxIM automatically!

# Sun May  7 19:28:50 2017  jpmorgen@snipe

# OK, the code below implements a class, cCamera, which I guess is for
# control of the camera.  Seems reasonable to add methods for guide
# control like I want them.


ERROR = True
NOERROR = False
LIGHT_PATH = r"C:\Users\jpmorgen\Desktop\IoIO\data"
#SETTLE_TIME = 120  # minimum time for stable temp at set point 
#SETTLE_MAX  = 480  # maximum time for temp to stabilize
SETTLE_TIME = 1  # minimum time for stable temp at set point 
SETTLE_MAX  = 5  # maximum time for temp to stabilize
##------------------------------------------------------------------------------
## Class: cCamera
##------------------------------------------------------------------------------
class cCamera:
    def __init__(self):
        self.__CCDTemp = 'Skip' # CCD Temperature (Skip = don't change temp)
                
        print("Connecting to MaxIm DL...")
        self.__CAMERA = win32com.client.Dispatch("MaxIm.CCDCamera")
        # Doesn't let MaxIm shut down after last client exits
        self.__CAMERA.DisableAutoShutdown = True
        try:
            self.__CAMERA.LinkEnabled = True
        except:
            print("... cannot connect to camera")
            print("--> Is camera hardware attached?")
            print("--> Is some other application already using camera hardware?")
            raise EnvironmentError('Halting program')
        if not self.__CAMERA.LinkEnabled:
            print("... camera link DID NOT TURN ON; CANNOT CONTINUE")
            raise EnvironmentError('Halting program')
        self.__guideStarXPos = 0 # x-coordinate of guide star
        self.__guideStarYPos = 0 # y-coordinate of guide star
        self.__guideExposure = 1.0 # default guide exposure in seconds
        self.__guideSettleLimit = 0.40 # max pixel error before imaging can occur
        self.__guideSettleMaxTime = 120 # max time for autoguider to settle

        self.guider_plate = 4.42
        self.guider_angle = 178+9.5/60 - 180
    

    def rot(vec, theta):
        """Rotates vector counterclockwise by theta degrees"""
        theta = np.radians(theta_deg)
        c, s = np.cos(theta), np.sin(theta)
        M = np.matrix([[c, -s], [s, c]])
        return(np.dot(M, vec))
    
    # Consider these as references
    # http://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
    # http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    def guiderMove(self, dra, ddec, dec=None):
        """Moves the telescope using guider slews.  dra, ddec in arcsec"""
        if dec is None:
            dec = self.__CAMERA.GuiderDeclination
        # Change to rectangular tangential coordinates for small deltas
        dra = dra*np.cos(dec)
        # The guider motion is calibrated in pixels per second, with
        # the guider angle applied separately.  We are just moving in
        # RA and DEC, so we don't need to worry about the guider angle
        dpix = [dra, ddec] * self.guider_plate
        # Multiply by speed, which is in pix/sec
        dt = dpix * [self.__CAMERA.GuiderXSpeed, self.__CAMERA.GuiderYSpeed]
        
        if dt[0] > 0:
            retval = self.__CAMERA.GuiderMove(0, dt[0]):
        if dt[0] < 0:
            retval = self.__CAMERA.GuiderMove(1, -dt[0])
        # Can't find property that records state of simultaneous
        # slews, so just do one direction at a time
        while self.__CAMERA.GuiderMoving:
            time.sleep(0.1)
        if dt[1] > 0:
            self.__CAMERA.GuiderMove(2, dt[1])
        if dt[1] < 0:
            self.__CAMERA.GuiderMove(3, -dt[1])
        while self.__CAMERA.GuiderMoving:
            time.sleep(0.1)
        
        ## Correct for guider misalignment
        #v = rot(v, self.__CAMERA.GuiderAngle)
        ## Change dra and ddec into seconds of motion.
        #
        #
        #self.rot([dra, ddec]
        #dra_pix = dra * 
        #np.dot(R_mat(self.__CAMERA.GuiderAngle), [dra_pix, ddec_pix])
        ## The guider speeds are calibrated in guider pixels per second
        #self.__CAMERA.GuiderYSpeed 
        #math.radians(dra) / math.cos(self.__CAMERA.GuiderAngle)
        #math.radians(ddec) / math.sin(self.__CAMERA.GuiderAngle)
        #if dra > 0:
        #    self.__CAMERA.GuiderMove
    

    def setCCDTemp(self,strtemp):
        if strtemp.upper() == 'SKIP':
            print("No CCD Cooling Specified")
            self.__CCDTemp = -99
            return NOERROR
        if strtemp.endswith("C"):
            try:
                flttemp = float(strtemp[:-1])
            except:
                print("ERROR: Specified CCD Temperature - Invalid format")
                return ERROR
        else:
            try:
                flttemp = float(strtemp)
            except:
                print("ERROR: Specified CCD Temperature - Invalid format")
                return ERROR
        self.__CCDTemp = flttemp
        return NOERROR
        
    def gotoCCDTemp(self):
        if self.__CCDTemp > -90:
            # set the CCD temperature set-point
            self.__CAMERA.TemperatureSetpoint = self.__CCDTemp
            print("CCD temperature setpoint: %0.2fC" % self.__CCDTemp)
            # make sure the cooler is on, just in case
            if not self.__CAMERA.CoolerOn:
                print("Turning CCD cooler on")
                self.__CAMERA.CoolerOn = True
            print("Waiting for CCD temperature to stabilize")
            started = time.time()
            cnt = 0
            # Check CCD temp to stabilize
            while cnt < SETTLE_TIME and (time.time() - started) < SETTLE_MAX:
                currentTemp = self.__CAMERA.Temperature
                if (currentTemp < self.__CCDTemp - 0.5 or
                    currentTemp > self.__CCDTemp + 0.5):
                    cnt = 0
                time.sleep(1)
                cnt += 1
            if cnt == SETTLE_TIME and (time.time() - started) < SETTLE_MAX:
                print("CCD Temperature Stable at %0.2fC" % self.__CAMERA.Temperature)
                try:
                    power = self.__CAMERA.CoolerPower
                    print("CCD Cooler Power: %d%%" % power)
                except:
                    print("CCD cooler power could not be read")
                    return NOERROR
                return NOERROR
            else:
                print("CCD Temperature Did Not Stabilize")
                return ERROR
        else:
            print("Skipping temp stabilization.")
            return NOERROR

    def warmCCD(self):
        if self.__CAMERA.CoolerOn:
            print("Starting to gradually warm CCD temperature to ambient")
            power = 100
            setTemp = self.__CAMERA.Temperature
            while power > 3:
                setTemp = setTemp + 5.0
                self.__CAMERA.TemperatureSetpoint = setTemp
                print("CCD temperature setpoint: %0.2fC" % setTemp)
                print("Waiting 2.5 minutes for temperature to rise")
                time.sleep(150)
                print("CCD Cooler Temp : %0.2fC" % self.__CAMERA.Temperature)
                try:
                    power = self.__CAMERA.CoolerPower
                    print("CCD Cooler Power: %d%%" % power)
                except:
                    print("CCD cooler power could not be read")
                    power = 0
            print("CCD warming complete. Turning CCD cooler off")
            self.__CAMERA.CoolerOn = False
        else:
            print("CCD Cooler is off. CCD warming not necessary.")

    def generateFilename(self,path,baseName):
        # path is the path to where the file will be saved
        baseName.replace(':', '_')      # colons become underscores
        baseName.replace(' ', '_')      # blanks become underscores
        baseName.replace('\\', '_')     # backslash becomes underscore
        # make sure the base filename has an '_' at the end
        if not baseName.endswith("_"):
            baseName = baseName + "_"
        # add 1 to use next available number
        seqMax = self.getSequenceNumber(path,baseName) 
        seqNext = seqMax + 1
        # NOTE MODIFICATION FROM ORIGINAL SCRIPT 4-digit sequence and .fits
        filename = "%s%04d.fits" % (os.path.join(path,baseName),seqNext)
        return filename

    def getSequenceNumber(self,path,baseName):
        # get a list of files in the image directory
        col = os.listdir(path)
        # Loop over these filenames and see if any match the basename
        retValue = 0
        for name in col:
            front = name[0:-9]
            back = name[-9:]
            if front == baseName:
                # baseName match found, now get sequence number for this file
                seqString = name[-9:-5]  # get last 4 chars of name (seq number)
                try:
                    seqInt = int(seqString)
                    if seqInt > retValue:
                        retValue = seqInt    # store greatest sequence number
                except:
                    pass
        return retValue
        
    def exposeLight(self,length,filterSlot, name):
        print("Exposing light frame...")
        self.__CAMERA.Expose(length,1,filterSlot)
        while not self.__CAMERA.ImageReady:
            time.sleep(1)
        print("Light frame exposure and download complete!")
        # save image
        filename = self.generateFilename(LIGHT_PATH,name)
        print("Saving light image -> %s" % filename)
        self.__CAMERA.SaveImage(filename)

    def exposeGuider(self,length,filterSlot=None):
        try:
            if not filterSlot is None:
                self.__CAMERA.GuiderFilter(filterSlot)
                print("Set guider filter to ", str(filterSlot))
            print("Exposing guider frame for ", str(length))
            self.__CAMERA.GuiderExpose(float(length))
            while self.checkGuiderRunning():
                time.sleep(0.1)
            print("...done")
            return(NOERROR)
        except:
            print("ERROR: could not get guider to work")
            return(ERROR)        

    def setFullFrame(self):
        self.__CAMERA.SetFullFrame()
        print("Camera set to full-frame mode")
        
    def setBinning(self,binmode):
        tup = (1,2,3)
        if binmode in tup:
            self.__CAMERA.BinX = binmode
            self.__CAMERA.BinY = binmode
            print("Camera binning set to %dx%d" % (binmode,binmode))
            return NOERROR
        else:
            print("ERROR: Invalid binning specified")
            return ERROR
            
    def autoGuide(self,autoGuideStar,exposure):
        if autoGuideStar:
            self.__CAMERA.GuiderAutoSelectStar = True
            if self.__guideStarYPos == 0 or self.__guideStarYPos == 0:
                self.__guideExposure = exposure
                if self.exposeGuider(self.__guideExposure):
                    return ERROR
                self.__guideStarXPos = self.__CAMERA.GuiderXStarPosition
                self.__guideStarYPos = self.__CAMERA.GuiderYStarPosition
                print
                print("Guider Setup:")
                print("Guider:                %s" % self.__CAMERA.GuiderName)
                print("Guide star selection:  Auto")
                print("Guide star exposure:   %0.2f" % self.__guideExposure)
                print("Aggressiveness X-Axis: %0.2f" % \
                       self.__CAMERA.GuiderAggressivenessX)
                print("Aggressiveness Y-Axis: %0.2f" % \
                       self.__CAMERA.GuiderAggressivenessY)
                print("Max Move X-Axis:       %0.2f" % self.__CAMERA.GuiderMaxMoveX)
                print("Max Move Y-Axis:       %0.2f" % self.__CAMERA.GuiderMaxMoveY)
                print("Min Move X-Axis:       %0.2f" % self.__CAMERA.GuiderMinMoveX)
                print("Min Move Y-Axis:       %0.2f" % self.__CAMERA.GuiderMinMoveY)
        else:
            self.__CAMERA.GuiderAutoSelectStar = False
            ## if necessary, set up the guider for autoguiding
            if self.__guideStarXPos == 0 or self.__guideStarYPos == 0:
                # prompt operator to manually select a guide star
                print()
                print(" *** INPUT NEEDED ***")
                print(" 1. In MaxIm, manually expose the guide camera.")
                print(" 2. Click on a guide star and enter a guide exposure value.")
                print(" 3. Verify that MaxIm correctly tracks on the guide star.")
                input(" 4. Press ENTER key when ready to proceed: ")
                self.__guideStarXPos = self.__CAMERA.GuiderXStarPosition
                self.__guideStarYPos = self.__CAMERA.GuiderYStarPosition
                print()
                exposure = input(" Enter Guide Star Exposure (sec): ")
                try:
                    self.__guideExposure = float(exposure)
                except:
                    print(" ERROR: Invalid input. Expecting float value...try again")
                    exposure = input(" Enter Guide Star Exposure: ")
                    try:
                        self.__guideExposure = float(exposure)
                    except:
                        print("ERROR: Invalid input for guide star exposure")
                        return ERROR
                print()
                print("Guider Setup:")
                print("Guider:                %s" % self.__CAMERA.GuiderName)
                print("Guide star selection:  Manual")
                print("Guide star exposure:   %0.2f" % self.__guideExposure)
                print("Aggressiveness X-Axis: %0.2f" % \
                       self.__CAMERA.GuiderAggressivenessX)
                print("Aggressiveness Y-Axis: %0.2f" % \
                       self.__CAMERA.GuiderAggressivenessY)
                print("Max Move X-Axis:       %0.2f" % self.__CAMERA.GuiderMaxMoveX)
                print("Max Move Y-Axis:       %0.2f" % self.__CAMERA.GuiderMaxMoveY)
                print("Min Move X-Axis:       %0.2f" % self.__CAMERA.GuiderMinMoveX)
                print("Min Move Y-Axis:       %0.2f" % self.__CAMERA.GuiderMinMoveY)
    
        self.__CAMERA.GuiderBinning = 1
        self.__CAMERA.GuiderSetStarPosition(self.__guideStarXPos,
                                            self.__guideStarYPos)
        print("Guider Declination = %d" % self.__CAMERA.GuiderDeclination)
        print("Tracking on guide star at X = %d, Y = %d" % \
               (self.__guideStarXPos,self.__guideStarYPos))
        # start autoguiding
        try:
            guideStatus = self.__CAMERA.GuiderTrack(self.__guideExposure)
        except:
            print("ERROR: While attempting to start the autoguider")
            return ERROR
        else:
            if guideStatus:
                print("Start autoguiding...")
            else:
                print("ERROR: Autoguider did not start successfully")
                return ERROR
        print("Waiting for guider to settle below %0.2f px (max wait %d sec)" % \
               (self.__guideSettleLimit,self.__guideSettleMaxTime))
        started = time.time()
        cnt = 0
        while True:
            if (time.time() - started) > self.__guideSettleMaxTime:
                print("ERROR: Guider not settled within the max allowable time")
                return ERROR
            if self.__CAMERA.GuiderNewMeasurement:
                recentErrorX = self.__CAMERA.GuiderXError
                recentErrorY = self.__CAMERA.GuiderYError
                # ignore the first reading
                if cnt != 0:
                    print("X-Error: %7.3f  Y-Error: %7.3f" % \
                           (recentErrorX,recentErrorY))
                    if (abs(recentErrorX) < self.__guideSettleLimit and
                        abs(recentErrorY) < self.__guideSettleLimit):
                        break
            cnt += 1
            time.sleep(0.5)
        return NOERROR
    
    def stopAutoGuide(self):
        try:
            self.__CAMERA.GuiderStop()
        except TypeError:
            print("Stop Autoguiding...")
            time.sleep(2)
        except:
            print("ERROR: Unexpected error while attempting to stop autoguider")
    
    def checkGuiderRunning(self):
        return self.__CAMERA.GuiderRunning
        
    def resetGuideStar(self):
        self.__guideStarXPos = 0
        self.__guideStarYPos = 0

##
##    END OF 'cCamera' Class
##

#if __name__ == "__main__":
#   # Create an instance of the cCamera class
#    testCamera = cCamera()
#
#    # Setup MaxIm DL to take a full frame image 
#    testCamera.setFullFrame()
#    # Setup binning for 2x2
#    if not testCamera.setBinning(2):
#        # Expose filter slot 2 (Blue) for 12.5 seconds
#        testCamera.exposeLight(12.5,2)
#    else:
#        print("Image not taken due to previous error")

#if __name__ == "__main__":
#
#    # Create an instance of the cCamera class
#    testCamera = cCamera()
#
#    # Setup MaxIm DL to take a full frame image 
#    testCamera.setFullFrame()
#    # Setup binning for 2x2
#    if not testCamera.setBinning(2):
#        for i in range(4):
#            # Expose filter slot 0 (Red) for 15 seconds
#            testCamera.exposeLight(1,0,'m51_R_2x2')
#    else:
#        print("Image not taken due to previous error")


#if __name__ == "__main__":
#
#    # Create an instance of the cCamera class
#    testCamera = cCamera()
#
#    # Setup Maxim DL to take a full frame image 
#    testCamera.setFullFrame()
#    # Setup binning for 1x1
#    testCamera.setBinning(1)
#
#    # Set CCD Temperature
#    testCamera.setCCDTemp('-20C')
#    # Goto CCD Temperature
#    testCamera.gotoCCDTemp()
# 
#    # Take 3 images
#    for i in range(3):
#        # Expose filter slot 1 (Green) for 10 seconds
#        testCamera.exposeLight(10,1,'m51_G')
#    
#    # Set CCD Temperature
#    testCamera.setCCDTemp('-15C')
#    # Goto CCD Temperature
#    testCamera.gotoCCDTemp()
# 
#    # Take 3 images
#    for i in range(3):
#        # Expose filter slot 1 (Green) for 10 seconds
#        testCamera.exposeLight(10,1,'m51_G')
#    
#    # Set CCD Temperature
#    testCamera.setCCDTemp('skip')
#    # Goto CCD Temperature
#    testCamera.gotoCCDTemp()
# 
#    # Take 3 images
#    for i in range(3):
#        # Expose filter slot 1 (Green) for 10 seconds
#        testCamera.exposeLight(1,1,'m51_G')
#    
#    # Warm the CCD to ambient
#    testCamera.warmCCD()

#if __name__ == "__main__":
#
#    # Create an instance of the cCamera class
#    testCamera = cCamera()
#
#    # Setup Maxim DL to take a full frame image 
#    testCamera.setFullFrame()
#    # Setup binning for 1x1
#    testCamera.setBinning(1)
#
#    # Set CCD Temperature
#    testCamera.setCCDTemp('-15C')
#    # Goto CCD Temperature
#    testCamera.gotoCCDTemp()
#    
#    # Start camera autoguiding with auto guide star select
#    # Guide exposure = 1.0 second
#    if testCamera.autoGuide(True,1.0):
#        testCamera.stopAutoGuide()
#    # Make sure autoguider is running
#    if testCamera.checkGuiderRunning():
#        # Take 3 images
#        for i in range(3):
#            # Expose filter slot 2 (Blue) for 10 seconds
#            testCamera.exposeLight(10,2,'m51_B')
#        # Stop autoguider after all images complete
#        testCamera.stopAutoGuide()    
#    else:
#        print("ERROR - Autoguider not running as expected")
#
#    # Reset guide star positions for next test
#    testCamera.resetGuideStar()
#    # Start camera autoguiding with auto guide star select
#    # Guide exposure = 2.5 seconds
#    if testCamera.autoGuide(True,2.5):
#        testCamera.stopAutoGuide()
#    # Make sure autoguider is running 
#    if testCamera.checkGuiderRunning():
#        # Take 3 images
#        for i in range(3):
#            # Expose filter slot 2 (Blue) for 20 seconds
#            testCamera.exposeLight(20,2,'m51_B')
#        # Stop autoguider after all images complete
#        testCamera.stopAutoGuide()    
#    else:
#        print("ERROR - Autoguider not running as expected")
#    
#    # Reset guide star positions for next test
#    testCamera.resetGuideStar()
#    # Start camera autoguiding with manual guide star select
#    if testCamera.autoGuide(False,0):
#        testCamera.stopAutoGuide()
#    if testCamera.checkGuiderRunning():
#        # Take 3 images
#        for i in range(3):
#            # Expose filter slot 2 (Blue) for 15 seconds
#            testCamera.exposeLight(15,2,'m51_B')
#        # Stop autoguider after all images complete
#        testCamera.stopAutoGuide()    
#    else:
#        print("ERROR - Autoguider not running as expected")
#
#    # Warm the CCD to ambient
#    testCamera.warmCCD()


