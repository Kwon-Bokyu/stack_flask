class RotationType:
    RT_WHD = 0
    RT_HWD = 1
    RT_HDW = 2
    RT_DHW = 3
    RT_DWH = 4
    RT_WDH = 5

    ALL = [RT_WHD, RT_HWD, RT_HDW, RT_DHW, RT_DWH, RT_WDH]
    # un upright or un updown
    Notupdown = [RT_WHD,RT_HWD]
 
class Axis:
    HEIGHT = 0
    DEPTH = 1
    WIDTH = 2

    ALL = [HEIGHT, DEPTH, WIDTH]
