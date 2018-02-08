# I am having trouble getting into the ASCOM namesspace from Python,
# so just keep track of the few things I use here.

algGermanPolar = 2
# These are GEM-specific
pierEast = 0
pierUnknown = -1
pierWest = 1

# Throw in some MaxIm names too --> Do these mean RA and DEC or
# calculated X and Y based on guide rates and angle?  If they are the
# nominal guide ports with +X being east, +Y being north, redefining
# them to other numbers would be a way to reflect someone re-wiring
# their guider, as seems possible from studying the MaxIm guider
# setting advanced tab pier flip options
gdPlusX = 0
gdMinusX = 1
gdPlusY = 2
gdMinusY = 3

# Event Code Name 	Mask Bit Value	Event Code 
 
ceExposureStarted		= 1	# 0	
ceExposureCompleted		= 2	# 1	
ceExposureAborted		= 4	# 2	
ceSequenceStarted		= 8	# 3	
ceSequenceCompleted		= 16	# 4	
ceSequenceAborted		= 32	# 5	
ceGuiderTrackingStarted		= 64	# 6	
ceGuiderTrackingStopped		= 128	# 7	
ceGuiderExposureCompleted	= 256	# 8	
ceGuiderExposureStarFaded	= 512	# 9	
ceGuideCorrectionStarted	= 1024	# 10	
ceGuideCorrectionCompleted	= 2048	# 11	
ceFilterWheelMoving		= 4096	# 12	
ceFilterWheelStopped		= 8192	# 13	
ceConnected			= 16384	# 14	
ceDisconnected			= 32768	# 15	
ceExposurePaused		= 65536	# 16	
ceExposureResumed		= 131072# 17	
ceExposureReadoutCompleted	= 262144# 18	
ceGuiderCalCompleted		= 524288# 19	
