# author: David Hurwitz
# started: 10/31/19
#
# convert a value in the range [0:1] to one of 4 color scales: red, green, blue, purple
# looking at color charts, it's probably easiest to have about a few color, an interpolate between them.

red_scale = []
red_scale.append((255, 200, 200))      # light red      0.0
red_scale.append((255,   0,   0))      # red            0.5
red_scale.append((150,   0,   0))      # dark red       1.0

green_scale = []
green_scale.append((255, 255, 100))    # light yellow   0.0
green_scale.append((  0, 255,   0))    # green          0.5
green_scale.append((  0, 100,   0))    # dark green     1.0

blue_scale = []
blue_scale.append((200, 200, 255))     # light blue     0.0
blue_scale.append((  0,   0, 255))     # blue           0.5
blue_scale.append((  0,   0, 100))     # dark blue      1.0

purple_scale = []
purple_scale.append((255, 200, 255))   # light purple   0.0
purple_scale.append((255,   0, 255))   # purple         0.5
purple_scale.append(( 75,   0, 150))   # dark purple    1.0

#-------------------------------------------------------------------------------
# same as get_color() except
# returns (r, g, b) where each is [0 : 1.0]
#-------------------------------------------------------------------------------
def get_color_scaled(val, red, green, blue, purple):
    color = get_color(val, red, green, blue, purple)
    new_color = (color[0]/255, color[1]/255, color[2]/255)
    return(new_color)

#-------------------------------------------------------------------------------
# given val, return a color on the red, green, blue, or purple scale.
# exactly one of red, green, blue, or purple should be true.
# returns (r, g, b) where each is [0 : 255]
#-------------------------------------------------------------------------------
def get_color(val, red, green, blue, purple):

    # pick the correct scale (red, blue, or purple
    assert(red ^ green ^ blue ^ purple)
    scale = purple_scale
    if red:    scale = red_scale
    if green:  scale = green_scale
    if blue:   scale = blue_scale

    NumFixedColors = len(scale)
    N = NumFixedColors - 1

    # make sure val is [0:1]
    val = min(val, 0.9999999)
    val = max(val, 0.0)

    # get the closest colors on the scale
    lo = scale[int(val*N)]
    hi = scale[int(val*N) + 1]

    # get the fractional change between lo and hi
    fraction = float(val*N) - int(val*N)

    r = lo[0] + (hi[0] - lo[0]) * fraction
    g = lo[1] + (hi[1] - lo[1]) * fraction
    b = lo[2] + (hi[2] - lo[2]) * fraction
    return((r,g,b))
