import numpy as np
import math
import warnings
import shapely
import pandas as pd
from typing import Union

def read_32bit_color(bit32_color):
    """
    此函数将32位颜色提取成10进制数,范围0-255
    RGBA
    """
    # 从 32 位颜色值中提取 alpha 通道
    alpha = (bit32_color >> 24) & 0xFF
    red = (bit32_color >> 16) & 0xFF  # 提取红色通道
    green = (bit32_color >> 8) & 0xFF  # 提取绿色通道
    blue = bit32_color & 0xFF  # 提取蓝色通道
    return red, green, blue, alpha

def create_32bit_color(r, g, b, a=255):
    """
    此函数接收红、绿、蓝和 alpha 通道的值，并将其组合成一个 32 位的颜色值。
    输入值范围均为0-255。
    """
    # 将输入的颜色通道组合成 32 位颜色值
    bit32_color = (a << 24) | (r << 16) | (g << 8) | b
    return bit32_color

Tools2D_interactable_type = Union[
            "Tools2D.Segment",
            "Tools2D.SegmentGroup",
            "Tools2D.Line",
            "Tools2D.LineGroup",
            "Tools2D.DirectedLine",
            "Tools2D.DirectedLineGroup"
        ]

class Tools2D:

    class Vector:
        """
        method:
            change_norm
            rotate
            to_numpy
            to_list
        property:
            norm
        """
        def __init__(self, point_b,point_a=(0, 0)):
            self.np_l = None
            self.np_a = None
            self.np_b = None
            if isinstance(point_a, np.ndarray):
                self.np_a = point_a
                x_a, y_a = point_a
            elif isinstance(point_a, (list, tuple)):
                x_a, y_a = point_a
            else:
                raise ValueError(f"不支持的对象point_a:{type(point_a)}:{point_a}")

            if isinstance(point_b, np.ndarray):
                self.np_b = point_b
                if not self.np_a:
                    self.np_a = np.array(point_a)
                self.np_l = self.np_b - self.np_a
                self.v_x, self.v_y = self.np_l
            elif isinstance(point_b, (list, tuple)):
                x_b, y_b = point_b
                self.v_x = x_b - x_a
                self.v_y = y_b - y_a
            else:
                raise ValueError(f"不支持的对象point_a:{type(point_b)}:{point_b}")

        def __str__(self):
            return f"Tools2D.Vector: ({self.v_x}, {self.v_y})"

        @property
        def norm(self):
            return math.hypot(self.v_x, self.v_y)

        def change_norm(self, norm):
            if self.v_x == 0:
                if self.v_y < 0:
                    # 负数情况
                    self.v_x, self.v_y = 0, -norm
                self.v_x, self.v_y = 0, norm

            if self.v_y == 0:
                if self.v_x < 0:
                    self.v_x, self.v_y = -norm, 0
                self.v_x, self.v_y = norm, 0

            multiple = norm / self.norm
            self.v_x, self.v_y = self.v_x * multiple, self.v_y * multiple
            self.np_l = None
            return self

        def rotate(self, theta):
            theta_rad = math.radians(theta)  # 将角度转换为弧度
            cos_value = math.cos(theta_rad)  # Use math.cos
            sin_value = math.sin(theta_rad)  # Use math.sin
            rotated_x = self.v_x * cos_value - self.v_y * sin_value
            rotated_y = self.v_x * sin_value + self.v_y * cos_value
            self.v_x, self.v_y = rotated_x, rotated_y
            self.np_l = None
            return self

        def to_numpy(self):
            if self.np_l is None:
                self.np_l = np.array([self.v_x,self.v_y])
            return self.np_l

        def to_list(self):
            return [self.v_x, self.v_y]

    class Point:
        def __init__(self,point:list|tuple|np.ndarray):
            if not isinstance(point,np.ndarray):
                self.point = np.array(point)
            else:
                self.point = point

        def to_list(self)->list:
            return self.point.tolist()

        def to_np(self)->np.ndarray:
            return self.point

    class Segment:
        """
        method:
            shift
        """
        def __init__(self, a_point:"Tools2D.Point", b_point:"Tools2D.Point"):
            self.a = a_point.to_np()
            self.b = b_point.to_np()
            self.s_np = None # self_ndarray_type
            self.t_l = None # self_Line_class_object


        def __str__(self):
            l1, l2 = self.to_list()
            return f'Tools2D.Segment: {l1}<-->{l2}'

        def shift(self, vector:"Tools2D.Vector"):
            vector = vector.to_numpy()
            self.a = self.a + vector
            self.b = self.b + vector
            self.s_np = None # self_ndarray_type
            self.t_l = None # self_Line_class_object
            return self

        def to_numpy(self):
            if self.s_np is None:
                self.s_np = np.array([self.a, self.b])
            return self.s_np

        def to_list(self):
            return [self.a.tolist(), self.b.tolist()]

        def to_line(self):
            if self.t_l is None:
                self.t_l = Tools2D.Line(point_a=self.a, point_b=self.b)
            return self.t_l

        def interaction(self, other:Tools2D_interactable_type):
            if isinstance(other,Tools2D.Segment):
                a_seg = shapely.LineString(self.to_numpy())
                b_seg = shapely.LineString(other.to_numpy())
                inter = a_seg.intersection(b_seg)
                # 检查交集类型
                if inter.is_empty:
                    return None  # 没有交点
                else:
                    # 如果交集是点，返回坐标
                    if inter.geom_type == 'Point':
                        return np.array([inter.x, inter.y])
                    elif inter.geom_type == 'LineString':
                        # 如果交集是线，处理线的情况
                        # 因为是有向线段，需要求两线段range交集
                        # 目前使用inf代表无穷大代替
                        return np.inf
                    else:
                        raise ValueError()
            #there is lots of conditions need to think
            #wait to continue...
            #SegmentGroup, Line, LineGroup, DirectLine, DirectLineGroup

        @property
        def range(self):
            points = self.to_numpy()
            x_range = [np.min(points[:, 0]), np.max(points[:, 0])]
            y_range = [np.min(points[:, 1]), np.max(points[:, 1])]
            return Tools2D.Range2D(x_range, y_range)

    class Range2D:
        """
        sometime range is like: [x1,x2] [y1,y2]
        this class is to treat is condition
        """
        def __init__(self,x_range,y_range):
            self.x_range = x_range
            self.y_range = y_range

    class SegmentGroup:

        def __init__(self, *args):
            self.sl_group = []
            for i in args:
                if isinstance(i, Tools2D.segment_group):
                    self.sl_group.extend(i.sl_group)
                elif isinstance(i, Tools2D.segment):
                    self.sl_group.append(i.to_numpy())
            self.sl_group = np.array(self.sl_group)

        def __add__(self, other):

            if isinstance(other, Tools2D.segment_group):
                self.sl_group = np.concatenate(self.sl_group, other.sl_group)
                return self

            if isinstance(other, Tools2D.segment):
                self.sl_group = np.append(self.sl_group, other.to_numpy())
                return self

            raise ValueError(f"加法不符合规范:{other}")

        def shift(self, vector):
            if isinstance(vector, Tools2D.vector):
                vector = vector
            else:
                vector = Tools2D.vector(vector).to_numpy()
            # 默认情况下，只要 A 的形状与 B 的最后一个轴匹配（例如都是 (2,)），广播就会成功。
            self.sl_group = self.sl_group + vector

        def to_lines(self):
            """
            批量将 sl_group 中的每一对点转换为直线对象 l
            使用 NumPy 向量化高效批量处理
            """
            # 提取所有点的坐标，假设 sl_group 中的每个元素是 (a_point, b_point)
            points = self.sl_group  # shape: (N, 2, 2) where N is the number of sl objects

            # 提取所有点的 x 和 y 坐标
            x1, y1 = points[:, 0, 0], points[:, 0, 1]  # 第一个点的 x 和 y
            x2, y2 = points[:, 1, 0], points[:, 1, 1]  # 第二个点的 x 和 y

            # 使用向量化的方式计算直线参数 a, b, c
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1

            # 将直线参数组成数组
            lines_params = np.vstack([a, b, c]).T
            # 结果的形状是 (N, 3)
            # +---------------------+
            # | [a[0], b[0], c[0]] | --> 第0条直线参数
            # | [a[1], b[1], c[1]] | --> 第1条直线参数
            # | [...             ] | --> ..n
            # +---------------------+
            l_group = Tools2D.line_group()
            l_group.lines = lines_params
            return l_group

        def to_numpy(self):
            return self.sl_group

        def to_list(self):
            return self.sl_group.tolist()

        def range_x(self):
            # 提取每个线段的 x 坐标，形状为 (N, 2)
            x_matrix = self.sl_group[:, :, 0]
            x_matrix = np.sort(x_matrix, axis=1)
            # 返回的矩阵:
            # +------------------+
            # | [min_x1, max_x1] |  --> 线段 1 的 x 范围
            # | [min_x2, max_x2] |  --> 线段 2 的 x 范围
            # | ...              |  --> ...
            # | [min_xN, max_xN] |  --> 线段 N 的 x 范围
            # +------------------+
            return x_matrix

        def interaction(self):
            #TODO line_group,line|segment_group,segment_line|direct_gourp,direct_line|
            raise

    class Line:
        """
        ax+by+c=0
        """
        def __init__(self, a=None, b=None, c=None, point_a=None, point_b=None, direct_point=None, *args):
            """
            初始化直线，可以通过以下方式：
            1. 直接指定 a, b, c
            2. 通过两点 point_a 和 point_b
            3. 通过一点 point_a 和方向向量 direct_point
            """
            # 情况 1：直接提供 a, b, c
            if a is not None and b is not None and c is not None:
                if a == 0 and b == 0:
                    raise ValueError("a 和 b 不能同时为 0，无法定义直线")
                self.a = float(a)
                self.b = float(b)
                self.c = float(c)

            # 情况 2：通过两点 point_a 和 point_b 定义直线
            elif point_a is not None and point_b is not None:
                # 转换为 NumPy 数组，确保输入是可计算的
                p1 = np.array(point_a, dtype=float)
                p2 = np.array(point_b, dtype=float)
                if np.array_equal(p1, p2):
                    raise ValueError("两点重合，无法定义直线")
                # 两点式推导 ax + by + c = 0
                # 设 point_a = (x1, y1), point_b = (x2, y2)
                # a = y1 - y2, b = x2 - x1, c = x1y2 - x2y1
                self.a = p1[1] - p2[1]  # y1 - y2
                self.b = p2[0] - p1[0]  # x2 - x1
                self.c = p1[0] * p2[1] - p2[0] * p1[1]  # x1*y2 - x2*y1
                if self.a == 0 and self.b == 0:
                    raise ValueError("计算结果无效，直线参数错误")

            # 情况 3：通过一点 point_a 和方向向量 direct_point 定义直线
            elif point_a is not None and direct_point is not None:
                # 转换为 NumPy 数组
                p = np.array(point_a, dtype=float)
                d = np.array(direct_point, dtype=float)
                if np.all(d == 0):
                    raise ValueError("方向向量不能为零向量")
                # 方向向量 (dx, dy) 的法向量是 (-dy, dx)
                # 直线方程：-dy(x - x0) + dx(y - y0) = 0
                self.a = -d[1]  # -dy
                self.b = d[0]  # dx
                self.c = d[1] * p[0] - d[0] * p[1]  # -dy*x0 + dx*y0
                if self.a == 0 and self.b == 0:
                    raise ValueError("计算结果无效，直线参数错误")

            else:
                raise ValueError("参数不足，请提供 a,b,c 或 point_a,point_b 或 point_a,direct_point")

        def __str__(self):
            """返回直线方程的字符串表示"""
            if self.k is not None:
                _b = self._kxb_b
                if _b>0:
                    return f'y={self.k}x+{_b}'
                elif _b<0:
                    return f'y={self.k}x{_b}'
                else:
                    return f'y={self.k}x'
            else:
                return f'x={-self.c/self.a}'

        @property
        def k(self):
            """计算斜率，如果垂直则返回 None"""
            if self.b == 0:
                return None  # 垂直线，斜率无穷大
            return -self.a / self.b
        @property
        def _kxb_b(self):
            if self.b == 0:
                return None  # 垂直线
            return  -self.c/self.b

        def intersects(self, other):
            """计算与另一条直线的交点，若平行则返回 None"""
            if not isinstance(other, Tools2D.line):
                raise TypeError(f"必须与另一条直线对象比较,当前:{other}")
            # 解线性方程组：a1x + b1y + c1 = 0 和 a2x + b2y + c2 = 0
            A = np.array([[self.a, self.b], [other.a, other.b]])
            B = np.array([-self.c, -other.c])
            det = np.linalg.det(A)  # 行列式
            if abs(det) < 1e-10:  # 平行或重合
                return None
            # 求解交点
            point = np.linalg.solve(A, B)
            return point

        def distance_to_point(self, point):
            """计算点到直线的距离"""
            p = np.array(point, dtype=float)
            # 点到直线距离公式：|ax0 + by0 + c| / sqrt(a^2 + b^2)
            numerator = abs(self.a * p[0] + self.b * p[1] + self.c)
            denominator = np.sqrt(self.a ** 2 + self.b ** 2)
            return numerator / denominator

        def get_x(self, y):
            if abs(self.a) < 1e-10:  # 检查 a 是否接近 0
                return None  # 水平线，x 无唯一解
            return -(self.b * y + self.c) / self.a

        def get_y(self, x):
            if abs(self.b) < 1e-10:  # 检查 b 是否接近 0
                return None  # 垂直线，y 无唯一解
            return -(self.a * x + self.c) / self.b

        def shift(self,vector):
            if isinstance(vector,Tools2D.vector):
                vector=vector.to_numpy()
            else:
                vector=Tools2D.vector(vector).to_numpy()
            dx, dy = vector
            self.c = self.c - self.a * dx - self.b * dy

        def to_sl(self, x_range=None, y_range=None):
            """
            x_range, y_range
            """
            if x_range is None and y_range is None:
                raise ValueError('没有范围range无法把line求解成线段')

            point = []
            # 排序输入的范围,防止错误
            if x_range is not None:
                x_min, x_max = sorted([x_range[0], x_range[1]])
                y_by_x_min, y_by_x_max = self.get_y(x_min), self.get_y(x_max)
                if y_by_x_min is not None and y_by_x_max is not None:
                    point.append([x_min, y_by_x_min])
                    point.append([x_max, y_by_x_max])

            if y_range is not None:
                y_min, y_max = sorted([y_range[0], y_range[1]])
                x_by_y_min, x_by_y_max = self.get_x(y_min), self.get_x(y_max)
                if x_by_y_min is not None and x_by_y_max is not None:
                    point.append([x_by_y_min, y_min])
                    point.append([x_by_y_max, y_max])

            point = np.array(point)
            point = np.sort(point, axis=0)

            # 计算中间两组点
            mid_index = len(point) // 2  # 假设 point 的长度是偶数
            mid_points = point[mid_index - 1:mid_index + 1]

            # 返回线段
            return Tools2D.segment(mid_points[0], mid_points[1])

        def to_numpy(self):
            return np.array([self.a,self.b,self.c])

    class LineGroup:
        def __init__(self,*args):
            self.lines = []
            for i in args:
                if isinstance(i, Tools2D.line_group):
                    self.lines.extend(i.lines)
                elif isinstance(i, Tools2D.line):
                    self.lines.append(i.to_numpy())
            self.lines = np.array(self.lines)

        def shift(self, vector):
            if isinstance(vector, Tools2D.vector):
                vector = vector.to_numpy()
            else:
                vector = Tools2D.vector(vector).to_numpy()
            dx, dy = vector
            self.lines[:, 2] = self.lines[:, 2] - self.lines[:, 0] * dx - self.lines[:, 1] * dy
            return self

        def interaction(self):
            # TODO line_group,line|segment_group,segment_line|direct_gourp,direct_line|
            raise

    class DirectedLine:
        def __init__(self,d_vector,l_point):
            raise
        def interaction(self):
            # TODO 这时返回需要分为(positive,negative)
            # TODO 可以使用Geo的Ray
            raise

    class DirectedLineGroup:
        def __init__(self,*args):
            raise

        def interaction(self):
            # TODO line_group,line|segment_group,segment_line|direct_gourp,direct_line|
            # TODO 这时返回需要分为(positive,negative)
            raise

    class Curve:
        def __init__(self,curve_type='Bezier',*args):
            if curve_type:
                raise
            raise
        def curve(self,*args):
            raise

    class Surface:
        def __int__(self,style='sketch'):
            raise

class Screen_draw:
    def __init__(self, py5):
        self.py5 = py5  # Store the py5 object
        self.tools = Tools2D()
        self.screen_info = self.screen_get_info()

    def screen_draw_surface(self, surfacedic, floor):
        self.tools.reset()
        surface_drawed = {}
        allsurfacelist = surfacedic.keys()
        for sf in allsurfacelist:
            thedic = surfacedic[sf]
            if thedic['floor'] != floor:
                continue
            weizhi = thedic['local']
            if not isinstance(weizhi, np.ndarray):
                weizhi = np.array(weizhi, dtype=float)
            vertices = weizhi
            surface_drawed[sf] = self.py5.create_shape()

            surface_drawed[sf].begin_shape()
            surface_drawed[sf].fill(thedic['color'])
            if thedic['stroke'] is None:
                surface_drawed[sf].no_stroke()
            else:
                surface_drawed[sf].stroke(thedic['stroke_color'])
            if not thedic['fill']:
                surface_drawed[sf].no_fill()
            surface_drawed[sf].fill(thedic['color'])
            surface_drawed[sf].vertices(vertices)
            surface_drawed[sf].end_shape()
            self.py5.shape(surface_drawed[sf])

    def screen_draw_SegmentLine(self, SegmentLine_dic, floor):
        self.tools.reset()
        for key, val in SegmentLine_dic.items():
            if val['floor'] != floor:
                continue
            if not val['visible']:
                continue
            color = val['color']
            self.py5.stroke(color)
            strokeweigh = val['stroke_weight']
            self.py5.stroke_weight(strokeweigh)
            local_group = []
            for each_point in val['location']:
                for i in each_point:
                    local_group.append(i)
            self.py5.line(*local_group)

    def screen_draw_vector(self, vector_or_vector_list, start_point):
        self.tools.reset()

        def arrow(vector):
            arrow_vector_A = self.tools.vector_rotate(vector, 180 - 30)
            arrow_vector_B = self.tools.vector_rotate(vector, -(180 - 30))
            arrow_vector_A = self.tools.vector_change_norm(arrow_vector_A, norm=10)
            arrow_vector_B = self.tools.vector_change_norm(arrow_vector_B, norm=10)
            arrow_end_A = self.tools.point_shift(arrow_vector_A, vector=vector)
            arrow_end_B = self.tools.point_shift(arrow_vector_B, vector=vector)
            return arrow_end_A, arrow_end_B

        segline_group = []
        if self.tools.list_depth(vector_or_vector_list) == 2:
            for each_vector in vector_or_vector_list:
                if each_vector[0] == 0 and each_vector[1] == 0:
                    continue
                for i in arrow(each_vector):
                    segline_group.append(
                        [self.tools.point_shift(each_vector, start_point), self.tools.point_shift(i, start_point)])
                segline_group.append([start_point, self.tools.point_shift(each_vector, start_point)])
        elif self.tools.list_depth(vector_or_vector_list) == 1:
            for i in arrow(vector_or_vector_list):
                segline_group.append([self.tools.point_shift(vector_or_vector_list, start_point),
                                      self.tools.point_shift(i, start_point)])
            segline_group.append([start_point, self.tools.point_shift(vector_or_vector_list, start_point)])

        for i in segline_group:
            self.tools.Segmentline_drop(i[0], i[1])
        self.screen_draw_SegmentLine(self.tools.get_Segmentline_dic(), floor=0)

    def draw_directed_line(self, line_detail_dict, color=create_32bit_color(10, 10, 0, 255), stroke_weight=3, floor=0,
                           minimum=50):
        self.tools.reset()
        input_value = {'color': color, 'stroke_weight': stroke_weight, 'floor': floor}
        x_range, y_range = self.screen_info['x_range'], self.screen_info['y_range']

        if 'directed' not in line_detail_dict.keys() or not line_detail_dict['directed']:
            raise ValueError(f"输入的有向直线有误{line_detail_dict}")
        #TODO 此处判断需要修改

        lx, ly = line_detail_dict['location_point']
        vx, vy = line_detail_dict['direction_vector']
        line_detail = self.tools.directed_line_to_line(line_detail_dict, temp=True)

        #按照屏幕范围换算成线段
        segment_line = self.tools.line_to_Segmentline(line_detail, x_range=x_range, y_range=y_range)
        if not segment_line:
            return False
        A_point, B_point = self.tools.Segmentline_get_info(segment_line)['location']
        self.tools.Segmentline_remove_by_chain(segment_line)

        Ax, Ay = A_point
        # Bx, By = B_point

        #判断A和B哪个是正方向的点
        vec_to_A = np.array([Ax - lx, Ay - ly])
        direction_vec = np.array([vx, vy])
        dot_product_A = np.dot(vec_to_A, direction_vec)
        if dot_product_A > 0:
            positive_point, negative_point = A_point, B_point
        else:
            positive_point, negative_point = B_point, A_point

        #在起点画点
        self.py5.stroke(color)
        self.py5.stroke_weight(stroke_weight + 4)
        self.py5.point(*line_detail_dict['location_point'])

        positive_sl = self.tools.Segmentline_drop(line_detail_dict['location_point'], positive_point,
                                                            **input_value)
        positive_sl_dict = self.tools.Segmentline_get_info(positive_sl)

        color_trans_segment_line_dicts = self._color_transition_segment_line(positive_sl_dict, **input_value,
                                                                             minimum=minimum)
        self.screen_draw(Seglinedic=color_trans_segment_line_dicts)

        negative_segment_line = self.tools.Segmentline_drop(negative_point, line_detail_dict['location_point'],
                                                            **input_value)
        negative_segment_line_dict = self.tools.Segmentline_get_info(negative_segment_line)
        dotted_negative_segment_line_dicts = self._dotted_segment_line(negative_segment_line_dict, spacing=10,
                                                                       **input_value)
        self.screen_draw(Seglinedic=dotted_negative_segment_line_dicts)
        return True

    def _color_transition_segment_line(self, seg_line_get_info, color, stroke_weight, floor=0, minimum=0, sampling=5):
        self.tools.reset()
        point_A, point_B = seg_line_get_info['location']
        x1, y1 = point_A
        x2, y2 = point_B

        color_rbga = list(read_32bit_color(color))
        delta_alpha = color_rbga[3] - minimum

        if color_rbga[3] <= minimum + sampling or delta_alpha <= sampling:
            warnings.warn(f"当前透明度:{color_rbga[3]},最小值{minimum},梯度小于采样值{sampling}")
            self.tools.Segmentline_drop(point_A, point_B, floor=floor, color=color, stroke_weight=stroke_weight)
            return self.tools.get_Segmentline_dic()

        line_num = math.ceil(delta_alpha / sampling)

        for i in range(0, line_num):
            ratio_color = i / (line_num - 1)
            the_color = color_rbga[:3] + [round(color_rbga[3] - delta_alpha * ratio_color)]
            ratio_start = i / line_num
            ratio_end = (i + 1) / line_num
            x_start = x1 + (x2 - x1) * ratio_start
            y_start = y1 + (y2 - y1) * ratio_start
            x_end = x1 + (x2 - x1) * ratio_end
            y_end = y1 + (y2 - y1) * ratio_end

            self.tools.Segmentline_drop([x_start, y_start], [x_end, y_end], floor=floor,
                                        color=self.py5.color(*the_color),
                                        stroke_weight=stroke_weight)
        return self.tools.get_Segmentline_dic()

    def _dotted_segment_line(self, seg_line_get_info, spacing, color, stroke_weight, floor=0):
        self.tools.reset()
        point_A, point_B = seg_line_get_info['location']
        x1, y1 = point_A
        x2, y2 = point_B

        line = self.tools.Segmentline_to_line([point_A, point_B], temp=True)
        a = line.get('a', 1)
        k = line.get('k', 0)
        total_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if a == 0:
            dx = 0
            dy = spacing * (1 if y2 > y1 else -1)
        else:
            dx = spacing / math.sqrt(1 + k ** 2)
            dy = k * dx
            if x2 < x1:
                dx = -dx
                dy = -dy

        num_segments = int(total_length // spacing)
        remainder = total_length % spacing

        if num_segments < 3:
            self.tools.Segmentline_drop([x1, y1], [x2, y2], floor=floor, color=color, stroke_weight=stroke_weight)
            return self.tools.get_Segmentline_dic()

        adjusted_spacing = spacing + (remainder / num_segments)

        if a == 0:
            dx = 0
            dy = adjusted_spacing * (1 if y2 > y1 else -1)
        else:
            dx = adjusted_spacing / math.sqrt(1 + k ** 2)
            dy = k * dx
            if x2 < x1:
                dx = -dx
                dy = -dy

        for i in range(num_segments):
            if i % 2 == 0:
                start_x = x1 + i * dx
                start_y = y1 + i * dy
                end_x = start_x + dx
                end_y = start_y + dy
                self.tools.Segmentline_drop([start_x, start_y], [end_x, end_y], floor=floor, color=color,
                                            stroke_weight=stroke_weight)
        return self.tools.get_Segmentline_dic()

    def screen_draw_lines(self, lines_dic, color=create_32bit_color(10, 10, 0, 255), stroke_weight=3):
        self.tools.reset()
        screen_info = self.screen_get_info()
        x_range, y_range = screen_info['x_range'], screen_info['y_range']

        for key, de_dic in lines_dic.items():
            self.tools.line_to_Segmentline(de_dic, x_range=x_range, y_range=y_range)

        self.py5.stroke(color)
        self.py5.stroke_weight(stroke_weight)

        line_to_draw = []
        for key, value in self.tools.Segmentline_dic.items():
            a_point, b_point = self.tools.Segmentline_get_info(key)['location']
            the_line = a_point + b_point
            line_to_draw.append(the_line)

        self.py5.lines(np.array(line_to_draw, dtype=np.float32))

    def screen_draw_directed_line(self, directed_line_dict_or_list, color=create_32bit_color(10, 10, 0, 255),
                                  stroke_weight=3):
        self.tools.reset()
        skip_times = 0 # ???
        if isinstance(directed_line_dict_or_list, dict): # ???
            lines = list(directed_line_dict_or_list.values())
        else:
            lines = directed_line_dict_or_list

        # print(len(lines))
        for line in lines:
            if not self.draw_directed_line(line, color=color, stroke_weight=stroke_weight):
                skip_times += 1

    def screen_draw_points(self, pointdic,s_weight=3, size=5, color=create_32bit_color(255, 0, 0, 255), fill=None):
        self.tools.reset()
        for key, value in pointdic.items():
            x, y = value
            self.py5.stroke_weight(s_weight)
            self.py5.stroke(color)
            if fill is None:
                self.py5.no_fill()
            else:
                self.py5.fill(fill)
            self.py5.circle(x, y, size)

    def screen_draw(self, f=3, Seglinedic=None, surfdic=None):
        self.tools.reset()
        if surfdic is None and Seglinedic is None:
            raise ValueError('没有输入surfdic或者seglinedic,无法绘制')
        for i in range(f):
            if surfdic is not None:
                self.screen_draw_surface(surfdic, i)
            if Seglinedic is not None:
                self.screen_draw_SegmentLine(Seglinedic, i)

    def screen_print_fps(self):
        self.py5.fill(0)
        self.py5.text_size(16)
        self.py5.text(f"FPS: {self.py5.get_frame_rate()}", 10, 30)

    def screen_get_info(self):
        in_dic = {}
        right_top = [self.py5.width, 0]
        right_down = [self.py5.width, self.py5.height]
        left_top = [0, 0]
        left_down = [0, self.py5.height]
        x_range = [0, self.py5.width]
        y_range = [0, self.py5.height]
        in_dic['x_range'] = x_range
        in_dic['y_range'] = y_range
        in_dic['rect'] = [left_top, right_top, right_down, left_down, left_top]
        in_dic['center'] = [self.py5.width / 2, self.py5.height / 2]
        return in_dic

    def screen_axis(self, x=0, y=0):
        x = self.py5.width / 2 + x
        y = self.py5.height / 2 - y
        return [x, y]


if __name__ == "__main__":
    t = Tools2D()
    seg_1 = t.Segment(t.Point([1, 0]), t.Point([0, 1]))
    seg_2 = t.Segment(t.Point([0, 0]), t.Point([1, 1]))
    print(seg_1.interaction(seg_2))



    # l = seg_2.to_line()
    # print(seg_1.to_line())
    # print(l.to_sl([0,2],[0.5,1]).to_numpy())
    # print(seg_1.interaction(seg_2))
