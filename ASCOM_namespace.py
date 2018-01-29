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
