import math
import py5_tools
py5_tools.add_jars('../jars')
import py5
from controlP5 import ControlP5

slider_dic = {}

def load():
    """
    初始化CP5对象
    """
    global cp5
    cp5 = ControlP5(py5.get_current_sketch())
    
def slider(title, location=(50,150), value=50, range_val=(0,50), size=(200,20)):
    global cp5
    global slider_dic
    value = math.floor(value)
    cp5.addSlider(title) \
        .setPosition(location[0], location[1]) \
        .setSize(size[0], size[1]) \
        .setRange(range_val[0], range_val[1]) \
        .setNumberOfTickMarks(range_val[1]-range_val[0]+1) \
        .setValue(math.floor(value)) \
        .setPosition(*location) \
        .setSize(*size) \
        .setRange(*range_val) \
        .setValue(value)
    slider_dic[title] = value

def slider_value():
    """
    如果发生改变slider_value字典
    反走返回None
    """
    global cp5
    global slider_dic
    for title in slider_dic.keys():
        the_siled = cp5.getController(title)
        if the_siled:
            now_value = math.floor(the_siled.getValue())
            if slider_dic[title] != now_value:
                slider_dic[title] = now_value
                #print(now_value)
                return slider_dic.copy()
    return None

