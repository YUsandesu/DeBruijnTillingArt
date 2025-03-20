import random
import BruijnsSystem
from the_control import *
from PY5_2DToolkit import *
import pandas as pd

start_location:list
def setup():
    global start_location
    py5.size(1280, 768)
    start_location= sd.screen_axis(0,0)
    load()
    slider('num1', [50, py5.height - 130], value=0, range_val=[0, 10],size=[400,30])
    slider('num2', [50, py5.height - 90], value=0, range_val=[0, 100], size=[400, 30])
    slider('num3', [50, py5.height - 50], value=0, range_val=[0, 1000], size=[400, 30])
    for k,v in tool.point_dic.items():
        tool.point_dic[k]=tool.point_shift(v,start_location)
    for k, v in p_dict.items():
        p_dict[k] = tool.point_shift(v, start_location)

def draw():
    global  tilling_data
    global p_dict
    py5.background(155)
    back = slider_value()
    if back is not None:
        i_seg_group = till.WFS(back['num1']+back['num2']+back['num3'])
        tool.reset()
        for seg in i_seg_group:
            tool.Segmentline_drop(seg[0], seg[1], color=py5.color(0, 0, 0, 100))
        for k, v in tool.point_dic.items():
            tool.point_dic[k] = tool.point_shift(v, start_location)

    sd.screen_draw_SegmentLine(tool.get_Segmentline_dic(),0)
    # sd.screen_draw_points(p_dict, size=13, color=py5.color(40, 40, 40, 255))
    sd.screen_draw_points(p_dict, size=60, color=py5.color(155))
    sd.screen_draw_points(p_dict, size=3, color=py5.color(40, 40, 40, 255))

if __name__ == "__main__":
    tool = Tools2D()
    a = BruijnsSystem.BruijnsSystem(sides=5,gap=13,shifted_distance=21,origin_norm=30,max_num_of_line=850)
    b = BruijnsSystem.BruijnsSystem(sides=5,gap=13,shifted_distance=11,origin_norm=30,max_num_of_line=850)
    #TODO 应该在在gird中把当前tilling的点画出来.
    print('Gird-finish')
    till = a.tilling
    till_b = b.tilling
    print('Tilling-finish')

    seg_group = till.WFS(500)
    print('WFS-finish')
    for i in seg_group:
        tool.Segmentline_drop(i[0],i[1],color=py5.color(120,120,120,255))

    p = list(tool.point_dic.keys())
    p = random.sample(p, k=120)

    seg_group =  till_b.WFS(500)
    for i in seg_group:
        tool.Segmentline_drop(i[0],i[1],color=py5.color(40,40,40,255))
    print(p)
    p_dict = {k:tool.point_dic[k] for k in p}
    sd = Screen_draw(py5)
    py5.run_sketch()
