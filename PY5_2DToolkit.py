import numpy as np
from collections import defaultdict
import math
import warnings
import shapely
import pandas as pd
from dask.dataframe.dispatch import tolist


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


class Tools2D:
    def __init__(self, screen_info=None):
        if screen_info:
            self.screeninfo = screen_info

    class vector:
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
            return f"({self.v_x}, {self.v_y})"

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

    class point:
        def __init__(self,point:list|tuple|np.ndarray):

            if not isinstance(point,np.ndarray):
                self.point = np.array(point)
            else:
                self.point = point

        def to_list(self):
            return self.point.tolist()

    class segment:

        def __init__(self, a_point, b_point):
            def to_np(v)->np.ndarray:
                if isinstance(v, Tools2D.point):
                    return v.point
                elif isinstance(v, np.ndarray):
                    return  v
                elif isinstance(v, (list, tuple)):
                    return np.array(v)
                raise ValueError(f"不支持的对象:{type(v)}:{v}")

            self.a = to_np(a_point)
            self.b = to_np(b_point)

            self.s_np, self.t_l, self.ran = [None] * 3

        def __str__(self):
            l1, l2 = self.to_list()
            return f'{l1}<-->{l2}'

        def shift(self, vector):
            if isinstance(vector,Tools2D.vector):
                vector = vector
            else:
                vector = Tools2D.vector(vector).to_numpy()

            self.a = self.a + vector
            self.b = self.b + vector
            self.s_np, self.t_l, self.ran = [None] * 3
            return self

        def to_numpy(self):
            if self.s_np is None:
                self.s_np = np.array([self.a, self.b])
            return self.s_np

        def to_list(self):
            return [self.a.tolist(), self.b.tolist()]

        def to_line(self):
            if self.t_l is None:
                self.t_l = Tools2D.line(point_a=self.a, point_b=self.b)
            return self.t_l

        def interaction(self,other):
            m = shapely.LineString(self.to_numpy())
            o = shapely.LineString(other.to_numpy())
            inter = m.intersection(o)
            # 检查交集类型
            if inter.is_empty:
                return None  # 没有交点
            else:
                # 如果交集是点，返回坐标
                if inter.geom_type == 'Point':
                    return np.array([inter.x, inter.y])
                elif inter.geom_type == 'LineString':
                    # 如果交集是线，处理线的情况
                    return None
                else:
                    return None

        @property
        def range_x(self):
            return np.sort(self.to_numpy()[:,0])

    class segment_group:

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

    class line:
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

    class line_group:
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

    class direct_line:
        def __init__(self,d_vector,l_point):
            raise
        def interaction(self):
            # TODO 这时返回需要分为(positive,negative)
            # TODO 可以使用Geo的Ray
            raise

    class direct_line_group:
        def __init__(self,*args):
            raise

        def interaction(self):
            # TODO line_group,line|segment_group,segment_line|direct_gourp,direct_line|
            # TODO 这时返回需要分为(positive,negative)
            raise

    class curve:
        def __init__(self,curve_type='Bezier',*args):
            if curve_type:
                raise
            raise
        def curve(self,*args):
            raise

    class surface:
        def __int__(self,style='sketch'):
            raise


    def distance_2_points_matrix(self, Apoint, Bpoint):
        """
        矩阵方法求norm 使用的是np矩阵
        """
        A = self.point_get_info(Apoint)['location']
        B = self.point_get_info(Bpoint)['location']
        np_A = np.array(A)
        np_B = np.array(B)
        return np.linalg.norm(np_A - np_B)

    def distance_2_points(self, point1, point2):
        """
        计算两点之间的距离，使用经典的平方根方法。

        参数:
        point1: 通用参数，可以是点的代号或[x, y]坐标。
        point2: 同上。

        返回:
        float: 两点之间的距离。

        异常:
        ValueError: 输入的点格式错误或不在字典中。
        """
        x1, y1 = self.point_get_info(point1)['location']
        x2, y2 = self.point_get_info(point2)['location']
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def vector_group_rotate_np(vector_group:list|np.ndarray, theta):
        """
        使用numpy方法旋转向量组
        对每一个数值应用reduce_errors_np

        参数:
            vector_group (np.ndarray | list[list]): 要旋转的向量组或单个向量。
                - NumPy数组,shape: (n, 2) 或 (2)
                - Python列表: [[x1, y1], ...] 或 [x, y]
            theta (float): 旋转角度（度）。正值为逆时针。

        返回:
            np.ndarray: 旋转后的向量组，形状与输入 `vector_group` 转换后的NumPy数组形状相同。

        ValueError: 输入 vector_group 格式错误（非二维向量或列表深度错误）。
        UserWarning: 输入为单个向量时发出警告。

        示例:
            Tools2D.vector_group_rotate_np(np.array([[1, 0], [0, 1]]), 45)
        """

        if isinstance(vector_group, np.ndarray):
            if vector_group.ndim == 1 and len(vector_group) == 2:
                #ndim:number of dimensions 实际取出来的是嵌套层数
                warnings.warn(f"输入的是一个单独向量: {vector_group}")
                vector_group = vector_group[np.newaxis, :]  # 将其转换为二维数组
            elif vector_group.shape[1] != 2:
                raise ValueError(f'输入的格式有误: {vector_group}')
        elif isinstance(vector_group, (list, tuple)):
            depth = Tools2D.list_depth(vector_group)
            if depth != 2:
                if depth == 1:
                    warnings.warn(f"输入的是一个单独向量:{vector_group}")
                    vector_group = [vector_group]
                else:
                    raise ValueError(f'输入的格式有误:{vector_group}')
            vector_group = np.array(vector_group)
        else:
            raise ValueError(f'输入的格式有误:{vector_group}')

        # 将角度转换为弧度
        theta = np.deg2rad(theta)

        # 旋转矩阵
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        rotated_vector = vector_group @ rotation_matrix.T  # 旋转向量

        # 处理接近0的浮点数
        rotated_vector = Tools2D.reduce_errors_np(rotated_vector, max_value=None)

        return rotated_vector

    @staticmethod
    def reduce_errors_np(nums: list | np.ndarray, min_value:float|None = 1e-10, max_value:float|None = 1e10):
        """
        减少ndarray,list,tuple中的数值误差
        1. 将绝对值小于min_value的值设置为零。
        2. 将绝对值大于max_value的值设置为 NaN（非数字）。

        Args:
           nums (list | np.ndarray): 要处理的目标
           min_value (None | float, 可选): 阈值，默认为 1e-10,低于此阈值（绝对值）的值被认为接近于零。
                                               如果设置为None，则跳过此最小值减少步骤。 默认为 1e-10。
           max_value (None | float, 可选): 同上,默认为 1e10

        Returns:np.ndarray

        Raises:
           ValueError: 输入nums不是list,tuple or ndarray

        """
        if isinstance(nums,np.ndarray):
            back_np = nums
        elif isinstance(nums,(list,tuple)):
            back_np = np.array(nums)
        else:
            raise ValueError (f"输入值类型错误:{nums},type:{type(nums)}")

        if min_value:
            back_np = np.where(np.abs(nums) < min_value, 0, back_np)
        if max_value:
            back_np = np.where(np.abs(nums) > max_value, np.nan, back_np)
        return back_np


    @staticmethod
    def reduce_errors(num, max_value=1e10, min_value=1e-10):
        """
        如果接近无穷大返回None，接近无穷小返回0
        """
        if abs(num) > max_value:
            return None
        elif abs(num) < min_value:
            return 0
        return num


    def vector_to_line(self, vector, passing_point=(0, 0), temp=False):
        """
        passing_point,直线经过的点，默认(0，0)
        向量换直线,返回的是字母代号。如果temp=True 返回字典。
        """
        if self.reduce_errors(vector[0]) == 0 and self.reduce_errors(vector[1]) == 0:
            return None
        if self.reduce_errors(vector[0]) == 0:
            # 垂直情况
            # x=b ; k=-1 a=0
            return self.line_drop(a=0, k=-1, b=passing_point[0])
        if self.reduce_errors(vector[1]) == 0:
            # 水平情况
            # y=b ;a=1 k=0
            return self.line_drop(a=1, k=0, b=passing_point[1])

        k = self.reduce_errors(vector[1] / vector[0])
        if k is None:
            a = 0  # 无穷小
        else:
            a = 1
        b = self.line_solve_general(a=a, x=passing_point[0], y=passing_point[1], k=k)['b']
        return self.line_drop(a=a, k=k, b=b, temp=temp)


    def line_shift(self, line_letter_or_dic, vector, rewrite=True, drop=True):
        """
            对line或directed_line进行平移操作。

            参数:
                line_letter_or_dic (str or dict): 直线的标识符（字符串）或直线的详细信息（字典）。
                vector (tuple or list): 平移向量[x,y]
                rewrite (bool): 是否更新 self.line_dic 中的直线信息。
                drop (bool): 当前直线还未创建,创建一个新的直线对象。
        """
        k = None
        if not isinstance(vector, (tuple, list)) or len(vector) != 2:
            raise ValueError(f"平移向量错误,当前为{vector}")

        if isinstance(line_letter_or_dic, str):
            detail = self.line_dic[line_letter_or_dic].copy()
            letter = line_letter_or_dic
        elif isinstance(line_letter_or_dic, dict):
            detail = line_letter_or_dic.copy()
            letter = None
        else:
            raise ValueError(f"输入直线错误,为{line_letter_or_dic}")

        if 'directed' in detail:  # 获取有向线段
            location_x, location_y = detail['location_point']
            detail['location_point'] = [location_x + vector[0], location_y + vector[1]]

        elif 'a' in detail:
            # x=b
            a = detail['a']
            k = -1
            new_b = detail['b'] + vector[0]
            detail['b'] = new_b
            detail['str'] = f'x={round(new_b, 2)}'
            if drop:
                return self.line_drop(k=k, b=new_b, a=a)

        else:  # 处理普通直线（y = kx + b 或水平线 y = b）
            b = detail['b']
            k = detail['k']

            if k == 0:  # 水平线（y = original_b）
                # y=b
                new_b = detail['b'] + vector[1]
                detail['str'] = f'y={round(new_b, 2)}'
            else:
                new_b = b + vector[1] - k * vector[0]  # (y-v1)=k(x-v0)+b-->y=kx - k*v0 + v1 +b -->b -k*v0 + v1
                if new_b > 0: detail['str'] = f'y={round(k, 2)}x+{round(new_b, 2)}'
                if new_b == 0: detail['str'] = f'y={round(k, 2)}x'
                if new_b < 0: detail['str'] = f'y={round(k, 2)}x{round(new_b, 2)}'
            detail['b'] = new_b

        if letter and rewrite:
            # 更新 self.line_dic 中的直线信息
            self.line_dic[letter] = detail
            return letter

        if drop:  # 返回新的直线对象
            if 'directed' in detail:
                return self.directed_line_drop(detail['location_point'], detail['direction_vector'])
            if k:
                return self.line_drop(k=detail.get('k'), b=detail['b'], a=detail.get('a', 1))

        else:  # 返回更新后的直线详细信息
            return detail


    # ////////////《线操作》////////////

    def directed_line_to_line(self, line_letter_or_detail_dic, temp=True):
        if isinstance(line_letter_or_detail_dic, dict):
            detail_dic = line_letter_or_detail_dic
        else:
            if line_letter_or_detail_dic not in self.line_dic:
                raise ValueError("没有找到直线，直线还未创建")
            detail_dic = self.line_dic[line_letter_or_detail_dic]

        if 'direction_vector' not in detail_dic or 'location_point' not in detail_dic:
            raise KeyError("有向直线的描述字典缺少必要的字段 'direction_vector' 或 'location_point'")

        vx, vy = detail_dic['direction_vector']
        lx, ly = detail_dic['location_point']
        if vx == 0:  # 垂直线
            # 垂直线公式：0y = -1x + b → x = b
            # 参数k在此场景下仅为占位符，固定为-1
            return self.line_drop(a=0, k=-1, b=lx, temp=temp)
        else:  # 常规线(包括水平)
            k = vy / vx
            b = ly - k * lx
            return self.line_drop(a=1, k=k, b=b, temp=temp)

    def _trans_line_to_matrix(self, line)-> np.ndarray:
        """
        将直线表示转换为矩阵形式（齐次坐标系下的直线方程系数）。

        Args:
            line: 可以是以下类型之一：
                - dict: 包含直线参数 {'k': 斜率, 'b': 截距}，可选 'a'（默认为 0）。
                - list: 包含多条直线表示（dict 或 id）的列表。
                - str/int/...: line_dic 中的直线 ID（不可变对象）。

        Returns:
            np.ndarray: 直线方程的系数矩阵 [k, -a, b]。
                - 对于单一输入，返回形状为 (3,) 的数组。
                - 对于列表输入，返回形状为 (N, 3) 的数组，其中 N 是直线数量。

        Raises:
            ValueError: 如果输入的 line ID 未在 line_dic 中找到。

        Notes:
            - 将直线方程 y=kx+b 表示为系数向量 [k, -a, b]。
            - 支持批量处理多条直线的转换。

        """
        def extract_np(l_d:dict)-> np.ndarray:
            if 'directed' not  in l_d:
                a = 0 if 'a' in l_d else 1
                k, b = l_d['k'], l_d['b']
                return np.array([k, -a, b])
            else:
                return extract_np(self.directed_line_to_line(l_d,temp=True))
        if isinstance(line, list):
            matrix_list = []
            for item in line:
                if isinstance(item, dict):
                    matrix_list.append(extract_np(item))
                else:
                    if item in self.line_dic:
                        matrix_list.append(extract_np(self.line_dic[item]))
                    else:
                        raise ValueError(f'line_dic中,没有找到id{item},完整输入:{line}')
            return np.array(matrix_list)  # Returns a NumPy array of matrices (stacked along axis 0)
        if isinstance(line, dict):
            return extract_np(line)
        if line in self.line_dic:
            return extract_np(self.line_dic[line])
        else:
            raise ValueError (f'line_dic中,没有找到id{line}')

    def inter_line_group_np(self,lines_a:list,lines_b:list, x_range: list | tuple = None, y_range: list | tuple = None):
        """

        计算两组直线的交点，使用 NumPy 批量处理。

        Args:
            lines_a (list): 第一组直线的列表，每个元素可以是：
                - dict: 包含直线参数 {'k': 斜率, 'b': 截距}，若包含 'a'，则 a=0，表示 0=kx+b
                - str/int/...: self.line_dic 中的直线 ID（不可变对象）
            lines_b (list): 第二组直线的列表，格式同 lines_a

        Returns:
            numpy.ndarray: 交点坐标矩阵，形状为 (N, M, 2)
                - N: lines_a 中的直线数量
                - M: lines_b 中的直线数量
                - 2: 每个交点的 [x, y] 坐标
                - 若交点不存在（平行线），对应位置值为 NaN

        Notes:
            - 使用齐次坐标系和叉积计算交点。
            - 利用 NumPy 的广播机制实现批量计算。
            - 输出矩阵的第 (i, j) 个元素表示 lines_a[i] 和 lines_b[j] 的交点。
            - 对平行线（w 接近零）的情况返回 NaN。
            - 使用阈值 1e-6 判断平行线。

        """
        a_np,b_np = self._trans_line_to_matrix(lines_a),self._trans_line_to_matrix(lines_b)
        # n_num = a_np.shape[0]
        # m_num = b_np.shape[0]
        # a_np: N * 3 | b_np: M * 3
        # 利用python-numpy中的广播机制 批量求解线段和直线的交点
        a_np = a_np[:, np.newaxis, :]  # N * 1 * 3
        b_np = b_np[np.newaxis, :, :]  # 1 * M * 3
        inter_homo = np.cross(a_np,b_np)# N * M * 3
        x, y, w = inter_homo[:, :, 0], inter_homo[:, :, 1], inter_homo[:, :, 2]

        # 计算掩码
        mask_no_inter = np.abs(w) > 1e-6  # 设置一个阈值来判断 w 是否接近零
        final_mask = mask_no_inter
        if x_range:
            x_min, x_max = sorted(x_range)
            mask_x = (x_min <= x) & (x <= x_max)
            final_mask = final_mask & mask_x
        if y_range:
            y_min, y_max = sorted(y_range)
            mask_y = (y_min <= y) & (y <= y_max)
            final_mask = final_mask & mask_y

        # 应用掩码计算最终结果
        inter_points_np = np.full_like(inter_homo[:, :, :2], [np.nan,np.nan], dtype=np.float64)
        # [:, :, :2] 索引切片 取[x,y] 原axis:2-->[x,y,w]
        inter_points_np[final_mask, 0] = self.reduce_errors_np(x[final_mask] / w[final_mask])  # 计算 x' = x / w
        inter_points_np[final_mask, 1] = self.reduce_errors_np(y[final_mask] / w[final_mask])  # 计算 y' = y / w
        return inter_points_np

    def intersection_2_Segmentline_Matrix(self, Aline, Bline):
        """
         使用矩阵方法 numpy 计算两条线段的交点
        :param Aline: 线段 A 的起点和终点坐标 [(x1, y1), (x2, y2)]
        :param Bline: 线段 B 的起点和终点坐标 [(x3, y3), (x4, y4)]
        :return: 交点坐标 (x, y)，如果没有交点返回 None
        """

        x1, y1 = Aline[0]
        x2, y2 = Aline[1]
        x3, y3 = Bline[0]
        x4, y4 = Bline[1]

        # 创建系数矩阵 A@缩小量=b
        A = np.array([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]])
        b = np.array([x3 - x1, y3 - y1])

        # 计算行列式
        det = np.linalg.det(A)

        # 判断是否平行或共线
        if abs(det) < 1e-10:  # 行列式接近 0，表示两条线段平行或共线
            return None

        # 解线性方程组
        t, s = np.linalg.solve(A, b)

        # 判断参数 t 和 s 是否在 [0, 1] 范围内
        if 0 <= t <= 1 and 0 <= s <= 1:
            # 计算交点坐标
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)

        return None  # 如果 t 或 s 不在范围内，则没有交点

    def intersection_2_Segmentline(self, A_seg_Chain_or_2pointxy, B_seg_Chain_or_2pointxy):
        """
        查找两条线段的交点，返回交点坐标或 None。
        支持混用链和点表示方式。

        参数:
        A_seg_Chain_or_2pointxy: 可以是链 (如 'A-B') 或两个点的列表 [[x, y], [x, y]]。
        B_seg_Chain_or_2pointxy: 可以是链或两个点的列表。

        返回:
        tuple: 交点坐标 (x, y) 或 None。

        异常:
        ValueError: 未找到线段或输入格式错误。
        """
        A_info = self.Segmentline_get_info(A_seg_Chain_or_2pointxy)
        if A_info is None:
            raise ValueError(f"没有找到A线段{A_seg_Chain_or_2pointxy}")
        Ax1, Ay1 = A_info['location'][0]
        Ax2, Ay2 = A_info['location'][1]
        B_info = self.Segmentline_get_info(B_seg_Chain_or_2pointxy)

        if B_info is None:
            raise ValueError(f"没有找到线段{B_seg_Chain_or_2pointxy}")
        Bx1, By1 = B_info['location'][0]
        Bx2, By2 = B_info['location'][1]

        # 特殊输入情况 防止报错
        if Bx1 == Bx2 and By1 == By2 and Ax1 == Ax2 and Ay1 == Ay2:
            # raise ValueError('输入了一个点')
            return Ax1, Ay1
        if Bx1 == Bx2 and By1 == By2:
            # raise ValueError('B线是一个点')
            temp_line_letter = self.Segmentline_to_line([A_seg_Chain_or_2pointxy[0], A_seg_Chain_or_2pointxy[1]])
            if By1 == self.line_solve(temp_line_letter, x=Bx1) and Bx1 == self.line_solve(temp_line_letter, y=By1):
                self.line_remove(temp_line_letter)
                return Bx1, By1
            else:
                self.line_remove(temp_line_letter)
                return None
        if Ax1 == Ax2 and Ay1 == Ay2:
            # raise ValueError('A线是一个点')
            temp_line_letter = self.Segmentline_to_line([B_seg_Chain_or_2pointxy[0], B_seg_Chain_or_2pointxy[1]])
            if Ay1 == self.line_solve(temp_line_letter, x=Ax1) and Ax1 == self.line_solve(temp_line_letter, y=Ay1):
                self.line_remove(temp_line_letter)
                return Ax1, Ay1
            else:
                self.line_remove(temp_line_letter)
                return None

        # 检查线段投影范围是否重叠（快速排除法）
        rangeX = max(min(Ax1, Ax2), min(Bx1, Bx2)), min(max(Ax1, Ax2), max(Bx1, Bx2))
        rangeY = max(min(Ay1, Ay2), min(By1, By2)), min(max(Ay1, Ay2), max(By1, By2))
        if rangeX[0] > rangeX[1] or rangeY[0] > rangeY[1]:
            return None  # 没有重叠，线段不可能相交

        # 计算直线的斜率和截距
        if Ax1 == Ax2:  # 第一条线和y轴水平
            k_A, b_A = None, Ax1
        else:
            k_A = (Ay1 - Ay2) / (Ax1 - Ax2)
            b_A = Ay1 - k_A * Ax1
        if Bx1 == Bx2:  # 第二条线和y轴水平
            k_B, b_B = None, Bx1
        else:
            k_B = (By1 - By2) / (Bx1 - Bx2)
            b_B = By1 - k_B * Bx1

        # 检查是否平行
        if k_A is None:  # 第一条线垂直
            x = Ax1
            y = k_B * x + b_B
        elif k_B is None:  # 第二条线垂直
            x = Bx1
            y = k_A * x + b_A
        else:
            if abs(k_A - k_B) < 1e-10:  # 斜率相等，平行，不可能有交点
                return None
            # 计算交点
            x = (b_B - b_A) / (k_A - k_B)
            y = k_A * x + b_A

        # 检查交点是否在两条线段的范围内
        if rangeX[0] <= x <= rangeX[1] and rangeY[0] <= y <= rangeY[1]:
            return x, y
        else:
            return None  # 交点不在线段范围内

    def intersection_line_and_Segmentline(self, segline_chain, line='a'):
        """
        经典方法 查找两条线段交点，无交点返回None
        """
        Aletter, Bletter = segline_chain.split('-')
        A = self.point_dic[Aletter]
        B = self.point_dic[Bletter]
        Ax, Ay = A
        Bx, By = B
        seg_rangeX, seg_rangeY = [min([Ax, Bx]), max([Ax, Bx])], [min([Ay, By]), max([Ay, By])]

        # 根据投影判断是否可能存在交点
        if not 'a' in self.line_dic[line]:
            if self.line_dic[line]['k'] != 0:
                line_shadow_y1 = self.line_dic[line]['k'] * seg_rangeX[0] + self.line_dic[line]['b']
                line_shadow_y2 = self.line_dic[line]['k'] * seg_rangeX[1] + self.line_dic[line]['b']
                line_shadow_Rangey = [min([line_shadow_y1, line_shadow_y2]), max([line_shadow_y1, line_shadow_y2])]
                line_shadow_x1 = (seg_rangeY[0] - self.line_dic[line]['b']) / self.line_dic[line]['k']
                line_shadow_x2 = (seg_rangeY[1] - self.line_dic[line]['b']) / self.line_dic[line]['k']
                line_shadow_Rangex = [min([line_shadow_x1, line_shadow_x2]), max([line_shadow_x1, line_shadow_x2])]
                final_range_x = [max(line_shadow_Rangex[0], seg_rangeX[0]), min(line_shadow_Rangex[1], seg_rangeX[1])]
                final_range_y = [max(line_shadow_Rangey[0], seg_rangeY[0]), min(line_shadow_Rangey[1], seg_rangeY[1])]
                if final_range_x[0] > final_range_x[1] or final_range_y[0] > final_range_y[1]:
                    # 范围无效 不存在交点
                    return None
                else:
                    if final_range_x[0] == final_range_x[1] and final_range_y[0] == final_range_y[1]:
                        # raise ValueError('范围仅为一个点')
                        print('范围仅为一个点')
                        letter_theline = self.Segmentline_to_line(segline_chain)
                        if self.line_solve(letter_theline, x=final_range_x[0]) == final_range_y[0]:
                            # 如果把点的x坐标带入直线中，得到的y值刚好是点的y坐标

                            return [final_range_x[0], final_range_y[0]]
                        else:
                            return None
                    temp_Ax, temp_Bx = final_range_x[0], final_range_x[1]
                    temp_Ay = self.line_solve(line, x=temp_Ax)
                    temp_By = self.line_solve(line, x=temp_Bx)
                    temp_A, temp_B = [temp_Ax, temp_Ay], [temp_Bx, temp_By]
                    inter_point = self.intersection_2_Segmentline([temp_A, temp_B], [A, B])
                    return inter_point
            else:
                # k=0时候，y=b 只需要比较线段的y范围是否包含b
                if seg_rangeY[0] <= self.line_dic[line]['b'] <= seg_rangeY[1]:
                    value_y = self.line_dic[line]['b']
                    the_line = self.Segmentline_to_line(segline_chain)
                    value_x = self.line_solve(the_line, y=value_y)
                    return [value_x, value_y]
                else:
                    return None
        else:
            if self.line_dic[line]['k'] != 0:
                raise ValueError("a=0 且 k=0 ：输入的是一个点而不是线")
            # a=0时候,x=b/-k 是一条垂直线 只需要比较线段的x范围是否包含b/-k
            if seg_rangeY[0] <= self.line_dic[line]['b'] / -self.line_dic[line]['k'] <= seg_rangeY[1]:
                value_x = self.line_dic[line]['b'] / -self.line_dic[line]['k']
                the_line = self.Segmentline_to_line(segline_chain)
                value_y = self.line_solve(the_line, x=value_x)
                return [value_x, value_y]
            else:
                return None

    def distance_point_to_line(self, point, line):
        # linedic例子:{'a': {'str': 'y=3x+100', 'k': 3, 'b': 100}}
        point_x = point[0]
        point_y = point[1]
        if isinstance(line, dict):
            detail_line_dic = line
        elif isinstance(line, str):
            detail_line_dic = self.line_dic[line]
        else:
            raise ValueError(f"输入的line不合符规范，为：{line}")

        if detail_line_dic['k'] == 0:
            # 输入的line是一条水平线
            return abs(detail_line_dic['b'] - point_y)
        if 'a' in detail_line_dic:
            # 输入的是一条垂直线
            # 存在a键的时候 a必定为0 且k必定为-1(line_drop中就是这么规定的)
            return abs(detail_line_dic['b'] - point_x)
        k_orth = -1 / detail_line_dic['k']
        # 斜率是-1/k的时候垂直
        result = self.line_solve_general(a=1, k=k_orth, x=point_x, y=point_y)
        b = result['b']
        line_orth_dic = self.line_drop(temp=True, k=k_orth, b=b, a=1)
        point = [point_x, point_y]
        point_inter = self.intersection_2line(line_orth_dic, detail_line_dic)
        return self.distance_2_points(point, point_inter)


    # ////////////《面操作》////////////
    def surface_drop_by_chain(self, chain_of_point, floor=0, color=create_32bit_color(200, 200, 20, 255), fill=False,
                              stroke=None,
                              stroke_color=create_32bit_color(0, 0, 0)):
        """
        【center】会自动生成在参数字典中：重心:是所有顶点坐标的平均值
        这里输入的链是不一定需要收尾相接的,如果不相接会自动补全
        """
        surf_pointgroup = []
        alist_of_point = chain_of_point.split('-')
        for aletter in alist_of_point:
            point_xy = self.point_dic.get(aletter, 0)
            if point_xy != 0:
                surf_pointgroup.append(point_xy)
            else:
                return "false:cant find point by letter"
        self.surface_chain_to_Segline_group(chain_of_point, visible=False)  # 确保线段都创建了
        nowdic = {}
        nowdic['floor'] = floor
        all_x, all_y = 0, 0
        for x, y in surf_pointgroup:
            all_x, all_y = all_x + x, all_y + y
        center = [all_x / len(surf_pointgroup), all_y / len(surf_pointgroup)]
        nowdic['center'] = center
        nowdic['local'] = surf_pointgroup
        nowdic['color'] = color
        nowdic['fill'] = fill
        nowdic['stroke'] = stroke
        nowdic['stroke_color'] = stroke_color
        self.surface_dic[chain_of_point] = nowdic
        return surf_pointgroup


    def surface_drop_by_pointlist(self, apointlist, floor=0, color=create_32bit_color(200, 200, 20, 255), fill=False,
                                  stroke=None,
                                  stroke_color=create_32bit_color(0, 0, 0)):
        """
        这里输入的链是不需要收尾相接的,如果不相接会自动补全
        """
        theletter = self.point_drop_group(apointlist)
        chain = "-".join(theletter)
        self.surface_drop_by_chain(chain, floor, color, fill, stroke, stroke_color)


    def surface_chain_to_Segline_group(self, chain, floor=0, color=create_32bit_color(0, 0, 0, 255), stroke_weight=3,
                                       visible=True):
        """
        给定一个字符串A-B-C,返回[A-B][B-C][C-A](返回的是首尾相接的,输入的不一定需要收尾相接)
        如果seglinedic中不存在这个线段 那么就会自动创建
        :param chain: 文本型，一个字符串 例：A-B-C
        :return: [A-B][B-C][C-A]
        """
        nodes = chain.split("-")  # 将链式结构分解为节点列表["A", "B", "C"]
        # 生成相邻对
        pairs = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        if nodes[0] != nodes[-1]:
            # 如果第一个点和最后一个点不一致,加入首尾连接
            pairs.append((nodes[-1], nodes[0]))  # 结果: [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]
        formatted_pairs = [f"{a}-{b}" for a, b in pairs]
        for i in formatted_pairs:
            # 检查是否已经创建了线段 如果不存在就创建线段
            if i in self.Segmentline_dic in self.Segmentline_dic:
                continue
            else:
                q = i.split('-')
                self.Segmentline_drop(q[0], q[1], floor=floor, color=color, stroke_weight=stroke_weight, visible=visible)
        return formatted_pairs


    def is_point_in_surface(self, polx, P):
        """
           polx接受列表型 也接受非齐次坐标矩阵
           判断点 P 是否在 polx 中（包括在边上）
           polx: 多边形的顶点矩阵 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
           P: 点的坐标 [x, y]
           return: 'inside' 如果点在内部, 'on_edge' 如果点在边上, 'outside' 如果点在外部
        """

        # 应当检查符号chain，给定的四边形是否已经闭合，若已经闭合 不可以用下文的
        # polx[(i + 1) % len(polx)]
        # 而应该改用
        # polx[i+1]
        def cross_product(A, B, P):
            # 计算叉积
            A = np.array(A)
            B = np.array(B)
            P = np.array(P)
            AB = B - A
            AP = P - A
            return np.cross(AB, AP)

        def is_point_on_segment(A, B, P):
            """
            判断点 P 是否在线段 AB 上
            :param A: 线段起点 [x1, y1]
            :param B: 线段终点 [x2, y2]
            :param P: 待检测点 [x, y]
            :return: True 如果 P 在线段 AB 上, 否则 False
            """
            # 叉积为 0 且点在线段范围内
            A = np.array(A)
            B = np.array(B)
            P = np.array(P)
            cross = cross_product(A, B, P)
            # 判断叉积是否为 0
            if abs(cross) > 1e-10:  # 允许微小误差
                return False
            # 判断是否在范围内
            dot_product = np.dot(P - A, B - A)  # 投影点是否在 A->B 的方向上
            squared_length = np.dot(B - A, B - A)  # AB 的平方长度
            return 0 <= dot_product <= squared_length

        # 检查每条边
        on_edge = False
        signs = []
        for i in range(len(polx)):
            A = polx[i]
            B = polx[(i + 1) % len(polx)]  # 四边形是闭合的
            if is_point_on_segment(A, B, P):  # 点在边上
                on_edge = True
            signs.append(cross_product(A, B, P))

        # 检查所有符号是否一致
        if all(s > 0 for s in signs) or all(s < 0 for s in signs):
            return 'inside' if not on_edge else 'on_edge'
        return 'on_edge' if on_edge else 'outside'


    def regular_polygon(self, sides, side_length):
        """。
        参数：
        - sides: 正多边形的边数。
        - side_length: 正多边形的边长。
        返回:
        多边形点的列表(顺时针方向)
        """

        def split_2pi(times):
            """
            将 2π 弧度 分成 times 等份。
            返回每份的弧度值。
            """
            return 2 * np.pi / times  # 360 度 = 2π 弧度

        if sides < 3:
            raise ValueError("边数必须大于或等于 3")
        if side_length <= 0:
            raise ValueError("边长必须大于 0")

        # 计算中心角的一半（theta / 2）
        theta = split_2pi(sides)  # 中心角的弧度值
        half_theta = theta / 2
        # 计算半径
        radius = side_length / 2 / np.sin(half_theta)
        point_start = [0, radius]  # 第一个点是从原点出发沿着y轴正方向前进的
        back_list = [point_start]
        for i in range(1, sides):
            point = self.vector_rotate(point_start, np.rad2deg(theta) * i)
            back_list.append(point)
        return back_list


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

def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"\033[31m{filename}:{lineno}: {category.__name__}: {message}\033[0m\n"
# 应用自定义格式
warnings.formatwarning = custom_formatwarning

if __name__ == "__main__":
    t = Tools2D()
    seg_1 = t.segment([1, 2], [2, 1])
    seg_2 = t.segment([1, 1], [2, 2])
    l = seg_2.to_line()
    print(seg_1.to_line())
    print(l.to_sl([0,2],[0.5,1]).to_numpy())
    print(seg_1.interaction(seg_2))
