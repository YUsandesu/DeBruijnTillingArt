import numpy as np
from collections import defaultdict
import math
import warnings

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
    """
    Tools2D 是一个为 2D 几何操作提供工具的实用类，包括点、直线段、直线、
    方向线和面的操作。它包含了创建、操作和查询各种 2D 几何实体的函数。

    属性:
        point_dic (dict): 存储点数据的字典。
        Segmentline_dic (dict): 存储线段数据的字典。
        surface_dic (dict): 存储面数据的字典。
        line_dic (dict): 存储线数据的字典。
        reverse_point_dic (defaultdict): 用于反向查找点的字典。
        alphabetize_Capital (list): 用于点/线命名的所有大写字母列表。
        alphabetize (list): 用于点/线命名的所有小写字母列表。
        letter_queue (list): 小写字母使用队列。
        letter_queue_capital (list): 大写字母使用队列。
        letter_index (int): 小写字母的编号索引。
        letter_index_capital (int): 大写字母的编号索引。
        screeninfo (dict 或 None): 屏幕信息，可选参数，用于默认取值范围。

    方法:
        reset():
            重新初始化对象到默认状态。
        get_point_dic():
            返回点的字典。
        get_Segmentline_dic():
            返回线段的详细字典。
        get_line_dic():
            返回线的字典。
        get_surface_dic():
            返回面的字典。
        point_drop(point_xy, specified=None):
            在指定的坐标处创建一个点，可选指定名字。
        distance_2_points_matrix(Apoint, Bpoint):
            使用 NumPy 返回两点之间的欧氏距离。
        distance_2_points(point1, point2):
            返回两点之间的欧氏距离。
        point_remove_by_letter(aletter):
            删除由字母标识的点。
        point_remove_by_xy(point_xy):
            删除由坐标标识的点。
        point_drop_group(point_xy_group):
            创建一组点。
        point_get_info(point_xy_or_letter):
            提供有关一个点的详细信息。
        point2_to_vector(Apoint, Bpoint):
            返回从 Apoint 到 Bpoint 的向量。
        point_shift(point_or_points, vector):
            将一个点或一组点按给定向量平移。
        vector_rotate(vector, theta):
            将提供的向量旋转 theta 角度。
        vector_get_norm(vector):
            返回向量的模（长度）。
        vector_change_norm(vector, norm=1):
            将向量的模（长度）改为指定值。
        vector_to_line(vector, passing_point=(0,0), temp=False):
            将向量转换成直线并可选存储。
        line_shift(line_letter_or_dic, vector, rewrite=True, drop=True):
            按照向量平移直线。
        Segmentline_drop(Apoint, Bpoint, floor=0, color=create_32bit_color(0, 0, 0, 255), stroke_weight=3, visible=True):
            在两点间创建线段。
        line_drop(k, b, a=1, temp=False):
            根据系数创建并可选存储一条线。
        line_remove(letter):
            删除一条线。
        line_to_Segmentline(line, x_range=None, y_range=None, floor=0, color=create_32bit_color(0, 0, 0, 255), stroke_weight=3, visible=True):
            将存储的线转换为线段。
        directed_line_to_line(line_letter_or_detail_dic, temp=True):
            将有向直线转换为普通直线。
        line_solve(line_letter_or_detail_dic, x=None, y=None):
            根据给定的 x 或 y 求解直线方程。
        line_solve_general(a=1, y=None, k=None, x=None, b=None, A_point=None, B_point=None):
            求解一般直线方程或根据两点确定直线。
        intersection_2_Segmentline_Matrix(Aline, Bline):
            使用矩阵方法查找两线段的交点。
        intersection_2_Segmentline(A_seg_Chain_or_2pointxy, B_seg_Chain_or_2pointxy):
            查找两条线段的交点。
        intersection_line_and_Segmentline(segline_chain, line='a'):
            查找一条线和一条线段的交点。
        intersection_2line(Aline_letter_or_kba_dic, Bline_letter_or_kba_dic):
            查找两条线的交点。
        Segmentline_shadow_on_axis(Chain_or_2pointxy):
            查找线段在 x 轴和 y 轴上的投影。
        Segmentline_to_line(chain_or_2pointxy, back_range=False, temp=False):
            将线段转换为直线。
        line_chain_or_dic(line_chain_or_dic):
            将输入规范化为字典格式。
        distance_point_to_line(point, line):
            查找点到线的距离。
        directed_line_drop(location_point=None, direction_vector=None, line_chain_or_dic=None):
            创建有向直线或将普通线转换为有向直线。
        line_to_directed_line(line_chain_or_dic, location_point):
            将普通线转换为有向直线。
        surface_drop_by_chain(chain_of_point, floor=0, color=create_32bit_color(200, 200, 20, 255), fill=False, stroke=None, stroke_color=create_32bit_color(0, 0, 0)):
            创建由链定义的面。
        surface_drop_by_pointlist(apointlist, floor=0, color=create_32bit_color(200, 200, 20, 255), fill=False, stroke=None, stroke_color=create_32bit_color(0, 0, 0)):
            创建由点定义的面。
        surface_chain_to_Segline_group(chain, floor=0, color=create_32bit_color(0, 0, 0, 255), stroke_weight=3, visible=True):
            将链转换为对应的线段。
        is_point_in_surface(polx, P):
            判断一个点是否在一个面内。
        regular_polygon(sides, side_length):
            生成正多边形的顶点。
        list_depth(lst):
            判断列表的嵌套深度。
        get_inter_range(a=None, b=None):
            查找两个区间的交集。
        clear_letter_mem_capital(used):
            清理大写字母内存。
        extract_letter_capital():
            从队列中提取一个大写字母。
        back_letter_capital(letter):
            将一个大写字母放回队列。
        apply_letter_capital(letter):
            申请一个指定的大写字母。
        clear_letter_mem(used):
            清理小写字母内存。
        extract_letter():
            从队列中提取一个小写字母。
        back_letter(letter):
            将一个小写字母放回队列。
        separate_letter(letter):
            将字母拆分为其基础 ASCII 值和索引。
    """

    def __init__(self, screen_info=None):
        self.point_dic = {}  # 存储点的字典
        self.Segmentline_dic = {}  # 存储线段的字典
        self.surface_dic = {}  # 存储面的字典
        self.line_dic = {}  # 存储直线的字典
        self.reverse_point_dic = defaultdict(list)  # 创建储存点的反字典 便于倒查
        # 初始化字母表
        self.alphabetize_Capital = [chr(i) for i in range(65, 91)]  # ASCII 65-90 对应 A-Z
        self.alphabetize = [chr(i) for i in range(97, 123)]  # ASCII 范围 97 到 122
        # 初始化字母队列和索引
        self.letter_queue = self.alphabetize.copy()  # [a,b,c...z]
        self.letter_queue_capital = self.alphabetize_Capital.copy()  # [A,B,C,D...Z]
        self.letter_index = 0  # 字母的后缀序列
        self.letter_index_capital = 0  # 字母的后缀序列
        if screen_info:
            self.screeninfo = screen_info

    def reset(self):
        self.point_dic.clear()  # 存储点的字典
        self.Segmentline_dic.clear()  # 存储线段的字典
        self.surface_dic.clear()  # 存储面的字典
        self.line_dic.clear() # 存储直线的字典
        self.reverse_point_dic = defaultdict(list)  # 创建储存点的反字典 便于倒查
        # 初始化字母表
        self.alphabetize_Capital = [chr(i) for i in range(65, 91)]  # ASCII 65-90 对应 A-Z
        self.alphabetize = [chr(i) for i in range(97, 123)]  # ASCII 范围 97 到 122
        # 初始化字母队列和索引
        self.letter_queue = self.alphabetize.copy()  # [a,b,c...z]
        self.letter_queue_capital = self.alphabetize_Capital.copy()  # [A,B,C,D...Z]
        self.letter_index = 0  # 字母的后缀序列
        self.letter_index_capital = 0  # 字母的后缀序列

    def get_point_dic(self):
        return self.point_dic

    def get_Segmentline_dic(self):
        back_dic = {}
        for i in self.Segmentline_dic.keys():
            back_dic[i] = self.Segmentline_get_info(i)
        return back_dic

    def get_line_dic(
            self):  # python has no public/private, public getter/setter functions (strong type) [variable; in a class, attribute -> property]
        """
        line_dic格式:{字母代号:{a:int,k:int,b:int}, 字母代号:{...}, ...}
        """
        return self.line_dic

    def get_surface_dic(self):
        return self.surface_dic

    # ================点线面存取操作================
    # ////////////《点操作》////////////

    def point_drop(self, point_xy, specified=None):
        """
        创建一个点并加入相应的字典中。

        参数:
        point_xy (list/tuple): 点的坐标，格式为 [x, y]。
        specified (str, 可选): 指定的点名称，如果为空则自动分配字母。

        返回:
        str: 新创建的点的字母代号。

        异常:
        ValueError: 输入坐标格式错误。
        ValueError: 指定点名称格式错误。
        """
        # 确认坐标格式正确
        if not self.list_depth(point_xy) == 1 and len(point_xy) == 2:
            raise ValueError(f"输入的坐标异常,为: {point_xy}")  # 输入格式应为[x, y]

        if specified is not None:
            if not isinstance(specified, str):
                raise ValueError(f"指定点名称格式错误,为: {specified}")
            letter = specified
            if specified in self.point_dic:
                # 如果当前指定的点已存在，则执行覆盖操作
                self.point_remove_by_letter(specified)
            self.apply_letter_capital(specified)  # 申请指定点
        else:
            letter = self.extract_letter_capital()
            # 如果未指定点，则自动分配一个点
        self.point_dic[letter] = point_xy  # 添加到字典
        self.reverse_point_dic[tuple(point_xy)].append(letter)  # 插入反向字典

        return letter

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

    def point_remove_by_letter(self, aletter):
        """
        删除指定字母对应的点，并将其重新插入到 nowletterlist 中。
        失败返回 False。
        """
        self.back_letter_capital(aletter)
        value = self.point_dic[aletter]
        del self.point_dic[aletter]
        if len(self.reverse_point_dic[tuple(value)]) > 1:
            # 如果列表中内容超过一项,那么有多个点名称同时存在
            re_write = self.reverse_point_dic[tuple(value)]
            re_write.remove(aletter)  # 从其中删除指定代号
            self.reverse_point_dic[tuple(value)] = re_write  # 重新覆写回去
        elif len(self.reverse_point_dic[tuple(value)]) == 1:
            del self.reverse_point_dic[tuple(value)]
        else:
            raise ValueError(f"删除reverse_pointdic时,发生未知错误altter:{aletter}")

    def point_remove_by_xy(self, point_xy):
        """
        通过调用point_get_info调用point_remove_by_letter
        失败返回False
        """
        info = self.point_get_info(point_xy)
        if info['letter'] is not None:
            return self.point_remove_by_letter(info['letter'])
        else:
            return False

    def point_drop_group(self, point_xy_group):
        """
        循环调用 point_drop
        :param point_xy_group: [a,b][c,d][e,f]格式不会会报错
        :return: 返回一个代号列表[A,B,C,D]
        """
        back = []
        for i in point_xy_group:
            if not isinstance(i, list):
                if not isinstance(i, tuple):
                    raise ValueError("data is not list or tuple")
            if len(i) == 2:
                back.append(self.point_drop(i))
            else:
                raise ValueError("data is not:([x,y],(x,y),(x,y))")
        return back

    def point_get_info(self, point_xy_or_letter):
        """
        如果还未创建,返回None
        无论输入的是 (x,y) or 字符代号
        统一返回一个字典 包括'type':返回'letter'或者'point_xy'
        'letter':字母代号 如果未找到此项会返回None
        'location':坐标值[x,y]
        """
        back_dict = {}
        if isinstance(point_xy_or_letter, str) and point_xy_or_letter:
            # 如果输入的是字母:
            if point_xy_or_letter not in self.point_dic.keys():
                # 输入的字母不在字典中
                raise ValueError(f"输入的字母{point_xy_or_letter}不在字典中:{self.point_dic}")
            detail_point = self.point_dic[point_xy_or_letter]
            point_x = detail_point[0]
            point_y = detail_point[1]
            letter = point_xy_or_letter
            input_type = 'letter'
        elif isinstance(point_xy_or_letter, (list, tuple, np.ndarray)) and len(point_xy_or_letter) == 2:
            point_x = point_xy_or_letter[0]
            point_y = point_xy_or_letter[1]
            input_type = 'point_xy'
            if tuple(point_xy_or_letter) in self.reverse_point_dic:
                letter = self.reverse_point_dic[tuple(point_xy_or_letter)][0]
            else:
                letter = None
        else:
            raise ValueError(f"未知错误,输入的点是{point_xy_or_letter}")
        back_dict['type'] = input_type
        back_dict['letter'] = letter
        back_dict['location'] = [point_x, point_y]
        return back_dict

    def point2_to_vector(self, Apoint, Bpoint):
        """
        默认是自A向B出发的向量
        point既可以输入字符代号,也可以输入[x,y]
        返回值:vector[x,y]
        """
        A_x, A_y = self.point_get_info(Apoint)['location']
        B_x, B_y = self.point_get_info(Bpoint)['location']
        vector = [self.reduce_errors(B_x - A_x), self.reduce_errors(B_y - A_y)]
        return vector

    @staticmethod
    def point_shift(point_or_points, vector):
        """
        将point按照vector的方向平移
        此方法point只接受[x,y],或者point的列表[[x,y],[x,y]]
        vector:[x,y]
        返回值:平移后新的point坐标[x,y]
        """
        if not isinstance(vector, (tuple, list)) or len(vector) != 2:
            raise ValueError(f"平移向量错误,当前为{vector}")
        s_x, s_y = vector
        if Tools2D.list_depth(point_or_points) == 2:
            # 如果输入的是一组点而不是一个点
            back_list = []
            for point in point_or_points:
                if len(point) != 2:
                    raise ValueError(f'point输入值有误:{point_or_points}')
                p_x, p_y = point
                back_x = p_x + s_x
                back_y = p_y + s_y
                back_list.append([back_x, back_y])
            return back_list
        if len(point_or_points) != 2:
            raise ValueError(f'point输入值有误:{point_or_points}')
        p_x, p_y = point_or_points[0], point_or_points[1]
        back_x = p_x + s_x
        back_y = p_y + s_y
        return [back_x, back_y]

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
    def vector_rotate(vector:list|np.ndarray|tuple, theta):
        """
        把向量按照theta(度数)旋转
        同时应用了reduce_errors
        参数:
            vector:list|ndarray|tuple 接受单个点
            theta:角度
        返回:
            新的向量:list
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        if isinstance(vector,(tuple,list)):
            if not Tools2D.list_depth(vector) == 1:
                raise ValueError(f"此方法只能操作单个点,当前输入:{vector}")
            elif len(vector) != 2:
                raise ValueError(f'此方法只能操作二维向量,当前输入:{vector}')

        theta_rad = math.radians(theta)  # 将角度转换为弧度
        cos_value = math.cos(theta_rad)  # Use math.cos
        sin_value = math.sin(theta_rad)  # Use math.sin

        x, y = vector  # Assumes vector is a list or tuple [x, y]

        rotated_x = x * cos_value - y * sin_value
        rotated_y = x * sin_value + y * cos_value

        # 防止无限接近0的情况
        return [Tools2D.reduce_errors(rotated_x), Tools2D.reduce_errors(rotated_y)]

    @staticmethod
    def vector_get_norm(vector):
        """
            计算向量的模长(长度)
            vector接受单个二维向量
            返回: 向量的模长, float
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        if isinstance(vector, (tuple, list)):
            if not Tools2D.list_depth(vector) == 1:
                raise ValueError(f"此方法只能操作单个向量,当前输入:{vector}")
            elif len(vector) != 2:
                raise ValueError(f'此方法只能操作二维向量,当前输入:{vector}')
        # math.hypot 函数可以正确处理负数
        back = math.hypot(vector[0], vector[1])
        return back

    @staticmethod
    def vector_change_norm(vector, norm=1):  # To numpy
        """
        调整向量的模长
        返回一个新的vector[x,y]
        """
        if not isinstance(vector, (tuple, list)) or len(vector) != 2:
            raise ValueError(f"平移向量错误,当前为{vector}")
        v_x, v_y = vector
        if v_x == 0 and v_y == 0:
            raise ValueError("无法修改0向量的模长")
        if v_x == 0:
            if v_y < 0:
                # 负数情况
                return [0, -norm]
            return [0, norm]
        if v_y == 0:
            if v_x < 0:
                return [-norm, 0]
            return [norm, 0]
        multiple = norm / math.hypot(v_x, v_y)  # normalization
        back = [v_x * multiple, v_y * multiple]
        return back


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
    def Segmentline_drop(self, Apoint, Bpoint, floor=0, color=create_32bit_color(0, 0, 0, 255), stroke_weight=3,
                         visible=True):
        """
        :param floor: 图层高度
        :param color: 只接受py5.color()之后的数值 否则后面绘制会出错
        :param visible: 是否可视，在绘制辅助线时候可以设置为=False
        """
        A_info = self.point_get_info(Apoint)
        if A_info['letter'] is None:
            Aletter = self.point_drop(A_info['location'])
        else:
            # 当前点已经存在 直接使用
            Aletter = A_info['letter']
        B_info = self.point_get_info(Bpoint)
        if B_info['letter'] is None:
            Bletter = self.point_drop(B_info['location'])
        else:
            # 当前点已经存在 直接使用"
            Bletter = B_info['letter']
        # 判断提供的点是否创建 如果没有创建就提前创建
        # print("#" * 10)
        # print(A_info, B_info) # ??? too many
        # print("#" * 10)
        inf = {}
        inf["floor"] = floor
        inf["color"] = color
        inf["stroke_weight"] = stroke_weight
        inf["visible"] = visible
        self.Segmentline_dic[Aletter + "-" + Bletter] = inf
        return Aletter + "-" + Bletter


    def Segmentline_get_info(self, chain_or_2pointxy):
        """
        如果返回None 说明该直线还未创建
        返回一个字典 包括键location[[x1,y1],[x2,y2]],chain,还有绘制信息(floor,color,visible,stroke_weight)
        """
        if isinstance(chain_or_2pointxy, str):
            if not chain_or_2pointxy in self.Segmentline_dic:
                return None
            A_point, B_point = chain_or_2pointxy.split('-')
        elif isinstance(chain_or_2pointxy, (list, tuple)) and len(chain_or_2pointxy) == 2:
            A_point = chain_or_2pointxy[0]
            B_point = chain_or_2pointxy[1]
        else:
            raise ValueError(f"输入的值有问题,为:{chain_or_2pointxy}")
        A_info, B_info = self.point_get_info(A_point), self.point_get_info(B_point)
        if A_info['letter'] is None or B_info['letter'] is None:
            return None
        back_dic = {}
        back_dic['location'] = [A_info['location'], B_info['location']]
        back_dic['chain'] = A_info['letter'] + '-' + B_info['letter']
        more = self.Segmentline_dic[back_dic['chain']]
        back_dic = back_dic | more
        return back_dic


    def Segmentline_remove_by_chain(self, chain):
        del self.Segmentline_dic[chain]


    def line_drop(self, k, b, a=1, temp=False):
        """
        保存一条直线方程 ay = kx + b。

        参数:
        k (float): 斜率。
        b (float): 截距。
        a (int): 常数项，默认为1。
        temp (bool): 如果为 True，则不存储到字典中，仅返回一个包含直线信息的字典。

        返回:
        str/字典: 直线的字母代号或包含直线信息的字典。

        异常:
        ValueError: a 和 k 不能同时为 0。
        """
        detail_dic = {}
        if a == 0 and k == 0:
            raise ValueError("a和k不能同时为0，请检查输入")
        if a == 0:
            line_str = f"x={round(b / -k, 2)}"
            detail_dic['str'] = line_str
            detail_dic['b'] = b / -k  # 0y=kx+b b/-k=x
            detail_dic['k'] = -1
            detail_dic['a'] = 0
        if k == 0:
            line_str = f"y={round(b, 2)}"
            detail_dic['str'] = line_str
            detail_dic['k'] = 0
            detail_dic['b'] = b
        if a == 1 and k != 0:
            if b > 0:
                line_str = f"y={round(k, 2)}x+{round(b, 2)}"
            elif b < 0:
                line_str = f"y={round(k, 2)}x{(round(b, 2))}"
            elif b == 0:
                line_str = f"y={round(k, 2)}x"
            else:
                raise ValueError(f"b值出现错误,b为:{b}")
            detail_dic['str'] = line_str
            detail_dic['k'] = k
            detail_dic['b'] = b
        if a != 0 and a != 1 and a is not None:
            raise ValueError(f"输入不符合要求,a只能是0或1,当前输入:{a}")
            # # ay+bx+c=0
            # k = b / a
            # b = k / a
            # if b > 0:
            #     line_str = f"y={k}x+{b}"
            # elif b < 0:
            #     line_str = f"y={k}x-{b}"
            # elif b == 0:
            #     line_str = f"y={k}x"
            #
            # detail_dic['str'] = line_str
            # detail_dic['k'] = k
            # detail_dic['b'] = b
        if temp:
            return detail_dic
        new_letter = self.extract_letter()
        self.line_dic[new_letter] = detail_dic
        return new_letter


    def line_remove(self, letter):
        """
        删除线
        失败返回False
        """
        if letter in self.line_dic:
            del self.line_dic[letter]
            self.back_letter(letter)
        else:
            return False


    def line_to_Segmentline(self, line, x_range=None, y_range=None, floor=0, color=create_32bit_color(0, 0, 0, 255),
                            stroke_weight=3,
                            visible=True):
        """
        line:可以接受 letter 或者 dict
        x_range,y_range:如果不提供默认使用屏幕尺寸
        """
        y_in_x_min, y_in_x_max = None, None
        x_min, x_max = None, None
        y_min, y_max = None, None

        input_value = {'floor': floor, 'color': color, 'stroke_weight': stroke_weight, 'visible': visible}

        # 判断输入是id
        if not isinstance(line, dict):
            if line not in self.line_dic:
                raise ValueError(f'{self.line_dic}没有找到给定line：{line}')
            line = self.line_dic[line]

        # 排序输入的范围,防止错误
        if x_range is not None:
            x_range = sorted([x_range[0], x_range[1]])
            x_min, x_max = x_range
        if y_range is not None:
            y_range = sorted([y_range[0], y_range[1]])
            y_min, y_max = y_range

        # 如果没有提供任何取值范围
        if x_range is None and y_range is None:
            print('没有提供 x_range 和 y_range 取值范围,使用屏幕范围')
            if self.screeninfo is None:
                raise ValueError("没有提供屏幕范围参数")
            x_range = self.screeninfo['xrange']
            y_range = self.screeninfo['yrange']


        # 如果没有提供完整取值范围,但是有屏幕信息
        elif x_range is None and self.screeninfo is not None:
            # 没有提供x_range取值范围,但可以使用屏幕范围的x_range缩小计算量
            x_range = self.screeninfo['xrange']
        elif y_range is None and self.screeninfo is not None:
            # 没有提供y_range取值范围,但可以使用屏幕范围的y_range缩小计算量
            y_range = self.screeninfo['yrange']

        value_k = line['k']
        value_b = line['b']

        # 如果是垂直线（ay=kx+b,其中a=0）
        if 'a' in line:

            if line['a'] != 0:
                raise ValueError(f'a应当仅为0或1，检查line：{line}')
            if y_range is None:
                raise ValueError(f'垂直线{line}没有y_range无法求解成线段')
            if y_min == y_max:
                raise ValueError(f'垂直线{line}取值范围为一个点，无法生成线段')

            # 0=kx+b 求解x的坐标
            value_x = value_b / -value_k

            if x_range is not None:
                if x_min <= value_x <= x_max:  # 如果提供了x范围,垂直线要在x的取值范围中
                    return self.Segmentline_drop(Apoint=[value_x, y_min],
                                                 Bpoint=[value_x, y_max],
                                                 **input_value)
                # warnings.warn(f'垂直线{line}不在x取值范围{x_range}')
                return None  # 不在范围中返回None
            else:
                return self.Segmentline_drop(Apoint=[value_x, y_min],
                                             Bpoint=[value_x, y_max],
                                             **input_value)
        elif x_min == x_max:
            # 上文已经判断过垂直线的情况
            warnings.warn(f'line:{line} x取值范围{x_range}是一个点')
            return None

        # 常规情况处理
        value_y = None
        if y_range:
            # 这里有可能解出任意值,None代表任意值,比如y刚好在水平线上
            x_solved_by_y_min = self.line_solve(line, y=y_min)
            x_solved_by_y_max = self.line_solve(line, y=y_max)

            if x_solved_by_y_min is None:
                # 水平线
                y_in_x_min, y_in_x_max = y_min, y_min
            elif x_solved_by_y_max is None:
                # if x_solved_by_y_min is None:
                #     raise ValueError(f'取得两个都是任意值,不太可能出现这种情况,line:{line},y_range:{y_range}')
                # 水平线
                y_in_x_min, y_in_x_max = y_max, y_max
            else:
                # 排序大小
                x_min_solved_by_y, x_max_solved_by_y = sorted([x_solved_by_y_min, x_solved_by_y_max])
                if x_range:
                    if x_max_solved_by_y <= x_min or x_min_solved_by_y >= x_max:
                        # print('超出范围')
                        return None
                    if x_min < x_min_solved_by_y < x_max:
                        x_min = x_min_solved_by_y
                    if x_min < x_max_solved_by_y < x_max:
                        x_max = x_max_solved_by_y
                else:
                    x_min, x_max = x_min_solved_by_y, x_max_solved_by_y
                # x_range = [x_min, x_max]

                # 如果是计算过的,直接使用缩小计算量
                # 此时不可能出现垂直线,因为上文已经判断过
                # 所以不会出现x_solved_by_y_min=x_solved_by_y_max
                if x_min == x_solved_by_y_min:
                    y_in_x_min = y_min
                elif x_min == x_solved_by_y_max:
                    y_in_x_min = y_max
                if x_max == x_solved_by_y_min:
                    y_in_x_max = y_min
                elif x_max == x_solved_by_y_max:
                    y_in_x_max = y_max

        # if x_range:
        if not y_in_x_min:
            y_in_x_min = self.line_solve(line, x=x_min)
        if not y_in_x_max:
            y_in_x_max = self.line_solve(line, x=x_max)
        # 这里求解不回None,因为输入x,返回任意值 说明刚好落在垂直线上,但是上文已经判断过a的情况

        return self.Segmentline_drop([x_min, y_in_x_min], [x_max, y_in_x_max])


    def directed_line_to_line(self, line_letter_or_detail_dic, temp=True):
        if isinstance(line_letter_or_detail_dic, dict):
            detail_dic = line_letter_or_detail_dic
        else:
            if line_letter_or_detail_dic not in self.line_dic: # ??? what is self.line_dic
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


    def line_solve(self, line_letter_or_detail_dic, x=None, y=None):
        """
        给定 x 或 y，解决直线方程 ay = kx + b 的问题。
        :param line_letter_or_detail_dic: 直线的标识字母（例：'a'），或者一个包含详细信息的字典。
        :param x: 已知的 x 值。
        :param y: 已知的 y 值。
        :return: 求解的另一个坐标值。
        如果返回None代表可以取任意值
        """
        # 获取直线的详细信息
        if isinstance(line_letter_or_detail_dic, dict):
            detail_dic = line_letter_or_detail_dic
        else:
            if line_letter_or_detail_dic not in self.line_dic:
                raise ValueError("没有找到直线，直线还未创建")
            detail_dic = self.line_dic[line_letter_or_detail_dic]

        # 检查是否提供了 x 或 y
        if x is None and y is None:
            raise ValueError("x 和 y 都没有输入值，无法计算")

        # 获取直线参数
        k = detail_dic['k']
        a = detail_dic.get('a', 1)  # 默认 a = 1
        b = detail_dic.get('b', 0)  # 默认 b = 0

        # 计算 y
        if y is None:
            if a == 0:
                return None  # 此时为垂直线,y可以取任意值
            return (k * x + b) / a  # y = (kx + b) / a

        # 计算 x
        if x is None:
            if k == 0:
                if a == 0:
                    raise ValueError("a = 0 且 k = 0 时，方程无意义，无法计算 x")
                return None  # 此时x取值为任意值
            return (a * y - b) / k  # x = (ay - b) / k


    def line_solve_general(self, a=1, y=None, k=None, x=None, b=None, A_point=None, B_point=None):
        """
        使用矩阵方法求解线性方程 ay = kx + b 或根据两点 (x1, y1) 和 (x2, y2) 计算 k 和 b。
        参数：
            a, y, k, x, b: 任意提供 ay = kx + b 的四个变量，求解第五个变量。
            A_point, B_point: 提供两点，求解 k 和 b，即使 x1 == x2 时也可以通过 a = 0 处理。
        返回：
            结果字典，包含已求解的变量及其值。
            当 a=0 且 k 都未给出时，函数将默认返回 k=1 ，即方程 x=-b
        """
        if a != 1 and a != 0:
            # 如果存在a值任意的话，把函数当作ay+kx+b=0来处理
            a = -a  # 我把这里添加了相反数是因为结尾输出的时候 是按照ay=kx+b书写的 所以-ay+kx+b=0
        # 1. 根据两点计算 k 和 b
        if A_point is not None and B_point is not None:
            if any(i is not None for i in (y, k, x, b)) and a != 1:
                raise ValueError("输入两点时候 请不要提供其它参数")
            if len(A_point) != 2 or len(B_point) != 2:
                raise ValueError("点格式不符合规范，A_point 和 B_point 应为 (x, y) 形式的元组或列表")
            x1, y1 = A_point
            x2, y2 = B_point
            if x1 == x2:  # 特殊情况：x1 == x2，垂直线
                return {
                    "a": 0,  # y 的系数为 0
                    "k": -1,  # 假设 k = -1
                    "b": x1,  # b = x1
                }
            else:  # 正常情况，使用矩阵方程计算 k 和 b
                A = np.array([[x1, 1], [x2, 1]])
                B = np.array([y1, y2])
                result = np.linalg.solve(A, B)
                return {"a": 1, "k": result[0], "b": result[1]}

        # 2. 特殊情况：a = 0
        if a == 0:
            # 方程退化为 kx + b = 0
            if k == 0:
                raise ValueError("0x+b=0")
            if k is not None and b is not None:
                return {"x": -b / k}
            elif x is not None:
                if k is None:
                    # 默认 k = 1
                    k = 1
                return {"b": -k * x}
            elif b is not None:
                if k is None:
                    # 默认 k = 1
                    k = 1
                return {"x": -b / k}
            else:
                raise ValueError("如果既没有 k，也没有 b，无解")

        # 3. 确保 y = kx + b 中有3个已知变量
        inputs = {"y": y, "k": k, "x": x, "b": b}
        known_values = {key: value for key, value in inputs.items() if value is not None}
        if len(known_values) != 3:
            raise ValueError("必须提供 y = kx + b 的3个变量才能求解")

        # 4. 根据未知变量求解，添加除零保护
        elif y is None:
            return {"y": (k * x + b) / a}
        elif k is None:
            if x == 0:
                raise ZeroDivisionError("x 不能为零")
            return {"k": (a * y - b) / x}
        elif x is None:
            if k == 0:
                raise ZeroDivisionError("k 不能为零")
            return {"x": (a * y - b) / k}
        elif b is None:
            return {"b": a * y - k * x}

        # 如果所有变量都已知，直接返回
        return known_values

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


    def intersection_2line(self, Aline_letter_or_kba_dic, Bline_letter_or_kba_dic):
        """
        :param Aline_letter_or_kba_dic: 可以是代号，也可以是save_line(temp=true)的返回值：一个包含k，b，a的字典
        :param Bline_letter_or_kba_dic: 可以是代号，也可以是save_line(temp=true)的返回值：一个包含k，b，a的字典
        返回值:[x,y]
        """

        # 判断是id还是dict
        if isinstance(Aline_letter_or_kba_dic, str):
            detail_dicA = self.line_dic[Aline_letter_or_kba_dic]
        else:
            detail_dicA = Aline_letter_or_kba_dic
        if isinstance(Bline_letter_or_kba_dic, str):
            detail_dicB = self.line_dic[Bline_letter_or_kba_dic]
        else:
            detail_dicB = Bline_letter_or_kba_dic

        # 判断是否为有向直线,转化为函数
        if 'directed' in detail_dicA and detail_dicA['directed'] is True:
            detail_dicA = self.directed_line_to_line(detail_dicA, temp=True)
        if 'directed' in detail_dicB and detail_dicB['directed'] is True:
            detail_dicB = self.directed_line_to_line(detail_dicB, temp=True)

        k_A = detail_dicA['k']
        b_A = detail_dicA['b']
        k_B = detail_dicB['k']
        b_B = detail_dicB['b']

        if 'a' in detail_dicA:
            if 'a' in detail_dicB:
                return None
            x = detail_dicA['b']  # 直接取b的值，而非b/k
            y = k_B * x + b_B
        elif 'a' in detail_dicB:
            x = detail_dicB['b']
            y = k_A * x + b_A
        elif k_A == k_B:
            return None
        else:
            x = (b_B - b_A) / (k_A - k_B)
            y = k_A * x + b_A

        return [x, y]


    def Segmentline_shadow_on_axis(self, Chain_or_2pointxy):
        """
        求一条线段分别在x轴和y轴的投影
        :param Chain_or_2pointxy: 既可以是A-B形式 也可以是[x,y][x,y]
        :return: 返回一个列表，包含两个范围[[x_min, x_max], [y_min, y_max]]
        """
        Seg_info = self.Segmentline_get_info(Chain_or_2pointxy)
        if Seg_info is None:
            raise ValueError(f"查找{Chain_or_2pointxy}失败")
        x1, y1 = Seg_info['location'][0]
        x2, y2 = Seg_info['location'][1]
        x_range = [min(x1, x2), max(x1, x2)]
        y_range = [min(y1, y2), max(y1, y2)]
        return [x_range, y_range]


    def Segmentline_to_line(self, chain_or_2pointxy, back_range=False, temp=False):
        """
        将由字符串 "A-B" 或者两个点的列表 [[x1, y1], [x2, y2]] 定义的线段转换为直线表示。

        :param chain_or_2pointxy: 可以是定义线段的字符串（例如 'A-B'），也可以是两个点的列表：[[Ax, Ay], [Bx, By]]。
        :param back_range: 如果为 True，返回额外的边界范围 [[xmin, xmax], [ymin, ymax]]。
        :param temp: 内部使用的临时参数（默认为 False）。
        :return: 表示直线的代号。
                 如果 back_range 为 True，返回一个包含 (line_code, bounding_box) 的元组。

        异常:
            ValueError: 如果定义的线段实际上是一个点，而不是线段。

        如果线段是垂直的，直线表示为 (0, x, -1)。
        如果线段是水平的，直线表示为 (1, y, 0)。
        否则，直线表示为 (1, 斜率, y_截距)。

        相关信息存储在 `line_dic` 字典中，以返回的代号为键。

        示例用法:
            line_code = obj.Segmentline_to_line('A-B')
            line_code, range = obj.Segmentline_to_line([[0, 0], [2, 2]], back_range=True)
        """

        if isinstance(chain_or_2pointxy, str):
            Aletter, Bletter = chain_or_2pointxy.split('-')
            x1, x2 = self.point_dic[Aletter][0], self.point_dic[Bletter][0]
            y1, y2 = self.point_dic[Aletter][1], self.point_dic[Bletter][1]
        else:
            x1, y1 = chain_or_2pointxy[0]
            x2, y2 = chain_or_2pointxy[1]

        if x1 == x2 and y1 == y2:
            raise ValueError('is not a line,this is a point')

        if x1 == x2:
            a = 0
            b = x1
            k = -1
        elif y1 == y2:
            a = 1
            k = 0
            b = y1
        else:
            a = 1
            k = (y1 - y2) / (x1 - x2)
            b = y1 - k * x1

        back = self.line_drop(k, b, a, temp)
        if back_range is True:
            the_range = self.Segmentline_shadow_on_axis(chain_or_2pointxy)
            back = back, the_range

        return back


    def line_chain_or_dic(self, line_chain_or_dic):
        """
        此方法无论输入的是dict还是'代号'
        统一会返回dict
        如果失败(输入的格式错误) 返回False
        """
        if isinstance(line_chain_or_dic, str):
            detail_line_dic = self.line_dic[line_chain_or_dic]
        elif isinstance(line_chain_or_dic, dict):
            detail_line_dic = line_chain_or_dic
        else:
            return False
        return detail_line_dic


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


    def directed_line_drop(self, location_point=None, direction_vector=None, line_chain_or_dic=None):
        """
        保存有向直线, 也可以将直线转换为有向直线
        location_point:起点
        direction_vector:方向向量
        此处使用的line_dic的字典中是ay=kx+b (a取值仅为0或1,当a=0时,字典中不存在a的键)
        """
        if location_point and len(location_point) != 2:
            raise ValueError("location_point必须是包含两个元素的列表或元组")
        if direction_vector and len(direction_vector) != 2:
            raise ValueError("direction_vector必须是包含两个元素的列表或元组")

        the_location_point = None
        the_direction_vector = None
        line_dict = None
        if line_chain_or_dic:
            line_dict = self.line_chain_or_dic(line_chain_or_dic)
            if not line_dict:
                raise ValueError(f"提供的Line错误{line_chain_or_dic}")
            if 'a' in line_dict:  # 垂直情况 a取值仅为0或1,如果为1就不存在键a
                the_direction_vector = [0, 1]
                the_location_point = [-line_dict['b'] / line_dict['k'], 0]
            else:
                the_direction_vector = [1, line_dict['k']]  # 常规情况

        if location_point:
            lo_x, lo_y = location_point
            if line_dict:  # 提供了起点,判断原点是否符合标准
                if self.line_solve(line_dict, lo_x) != lo_y:
                    raise ValueError(f"提供的起点:{location_point}不在直线{line_dict}上")
            the_location_point = location_point

        if direction_vector:
            dr_x, dr_y = direction_vector
            if line_dict:
                if line_dict['k'] != 0 and (dr_y == 0 or not math.isclose(dr_x / dr_y, line_dict['k'])):
                    raise ValueError(f"提供的方向{direction_vector}和直线{line_dict}的斜率不匹配")
                if line_dict['k'] == 0 and dr_y != 0:
                    raise ValueError(f"提供的方向{direction_vector}和水平直线{line_dict}不匹配")
            the_direction_vector = direction_vector

        if the_direction_vector and the_direction_vector:
            detail_dic = {
                'directed': True, 'location_point': the_location_point,
                'direction_vector': the_direction_vector
            }
            new_letter = self.extract_letter()
            self.line_dic[new_letter] = detail_dic
            return new_letter

        raise ValueError(
            f'缺少必要参数: location_point={location_point}, '
            f'direction_vector={direction_vector},'
            f' line_chain_or_dic={line_chain_or_dic}'
        )


    def line_to_directed_line(self, line_chain_or_dic, location_point):
        return self.directed_line_drop(location_point=location_point, line_chain_or_dic=line_chain_or_dic)


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


    # ////////////《常用操作》////////////
    @staticmethod
    def list_depth(lst):
        """
        静态方法
        """
        if not isinstance(lst, (list, tuple)):
            # 如果当前不是列表，层数为 0
            return 0
        if not lst:
            # 如果列表是空的，层数为 1（只有一层）
            return 1
        # 递归判断每个元素的嵌套深度，并取最大值
        return 1 + max(Tools2D.list_depth(item) for item in lst)


    def get_inter_range(self, a=None, b=None):
        """
        查找a和b的交集
        a和b的格式为[x,y]的范围
        无交集返回None
        """

        def get_range(interval):
            # 提取范围的辅助函数
            return min(interval[0], interval[1]), max(interval[0], interval[1])

        if a is not None and b is not None:
            if len(a) == 2 and len(b) == 2:
                amin, amax = get_range(a)
                bmin, bmax = get_range(b)
                new_max = min(amax, bmax)
                new_min = max(amin, bmin)
                # 检查是否有交集
                if new_min <= new_max:
                    return [new_min, new_max]
                else:
                    return None  # 无交集
            else:
                raise ValueError(f"输入{a}或{b}不是[x,y]形式")
        elif a is not None:
            if len(a) == 2:
                return list(get_range(a))
            else:
                raise ValueError(f"输入{a}不是[x,y]形式")
        elif b is not None:
            if len(b) == 2:
                return list(get_range(b))
            else:
                raise ValueError(f"输入{b}不是[x,y]形式")
        else:
            return None  # a 和 b 都为 None


    def clear_letter_mem_capital(self, used):
        """
        清理内存用
        输入已经使用的点的字母代号
        返回一个字典 每个代号对应一个新的名字
        """
        back_dic = {}
        if len(self.letter_queue) < len(used) * 2:
            return None
        for i in used:
            self.back_letter_capital(i)
        new_letter_index = []
        for i in used:
            new_letter = self.extract_letter_capital()
            new_letter_index.append(self.separate_letter(new_letter)[1])
            back_dic[i] = new_letter
        the_max_index = max(new_letter_index)  # 例如最大的是Z100 那么返回100
        for l, the_letter in enumerate(self.letter_queue_capital):
            the_ascii, the_index = self.separate_letter(the_letter)
            if the_index > the_max_index and the_ascii == ord('A'):
                # 如果是A开头的 例如当前遍历A100 我们的max是99 那么就通过
                # 当前遍历B99 我们的max是99 那么因为不是A开头的就会跳过 并且99<100
                del self.letter_queue_capital[l:]
                self.letter_index_capital = the_index - 1
                break
        return back_dic


    def extract_letter_capital(self):
        if not self.letter_queue_capital:
            # 如果字母队列中不够用了
            self.letter_queue_capital.extend([l + str(self.letter_index_capital + 1) for l in self.alphabetize_Capital])
            self.letter_index_capital += 1
        return self.letter_queue_capital.pop(0)  # 删除队列中的第一项并返回


    def back_letter_capital(self, letter):
        the_letter_ascii, the_letter_index = self.separate_letter(letter)
        for list_i, i in enumerate(self.letter_queue_capital):
            i_ascii, i_index = self.separate_letter(i)
            if i_index < the_letter_index:
                # 输入A100 当前查找到A51 列表为[A50,A51,...,A99]
                continue  # 此时应该跳过,寻找下一个
            if i_index == the_letter_index:
                # 输入F10 当前查找到A10 列表为[A10,B10,C10,D10]
                if i_ascii < the_letter_ascii:
                    continue
                if i_ascii > the_letter_ascii:  # 输入F10 当前查找为G10 那么应该插入在G前面
                    self.letter_queue_capital.insert(list_i, letter)  # 当前索引为i_index 插入会把当前项往后推移
                    return
            if i_index > the_letter_index:  # 当前输入A10 列表为[A12,A13,A14...]
                self.letter_queue_capital.insert(list_i, letter)  # 此时插入在当前项(之前)就可以
                return
        # 如果能到达此处 输入A100 当前查找到A51 列表为[A50,A51,...,A99]应该加入到末尾
        self.letter_queue_capital.append(letter)


    def apply_letter_capital(self, letter):
        """
        申请一个指定字母,并从字母表中删除它,成功返回True
        """
        the_ascii, the_index = self.separate_letter(letter)
        if the_index > self.letter_index_capital:
            # 队列不足开始创建
            for i in range(the_index - self.letter_index_capital):
                # 例如输入 A5 当前序号为3:[B3,C3....Z3]在添加两个循环(5-3) 得到[B3...Z5]
                self.letter_queue_capital.extend(
                    [l + str(self.letter_index_capital + 1) for l in self.alphabetize_Capital])
                self.letter_index_capital += 1
        if letter in self.letter_queue_capital:
            self.letter_queue_capital.remove(letter)
            return True
        raise ValueError(f"发生错误,队列为{self.letter_queue_capital},输入值为{letter}")


    def clear_letter_mem(self, used):
        back_dic = {}
        if len(self.letter_queue) < len(used) * 2:
            return None
        for i in used:
            self.back_letter(i)
        new_letter_index = []
        for i in used:
            new_letter = self.extract_letter()
            new_letter_index.append(self.separate_letter(new_letter)[1])
            back_dic[i] = new_letter
        the_max_index = max(new_letter_index)  # 例如最大的是Z100 那么返回100
        for l, the_letter in enumerate(self.letter_queue):
            the_ascii, the_index = self.separate_letter(the_letter)
            if the_index > the_max_index and the_ascii == ord('a'):
                # 如果是A开头的 例如当前遍历A100 我们的max是99 那么就通过
                # 当前遍历B99 我们的max是99 那么因为不是A开头的就会跳过 并且99<100
                del self.letter_queue[l:]
                self.letter_index = the_index - 1
                break
        return back_dic


    def extract_letter(self):
        """
        返回提取的字母
        同时从字母表中删除
        """
        if not self.letter_queue:
            # 如果字母队列中不够用了
            self.letter_queue.extend([l + str(self.letter_index + 1) for l in self.alphabetize])
            self.letter_index += 1
        return self.letter_queue.pop(0)  # 删除队列中的第一项并返回


    def back_letter(self, letter):
        the_letter_ascii, the_letter_index = self.separate_letter(letter)
        for list_i, i in enumerate(self.letter_queue):
            i_ascii, i_index = self.separate_letter(i)
            if i_index < the_letter_index:
                # 输入A100 当前查找到A51 列表为[A50,A51,...,A99]
                continue  # 此时应该跳过,寻找下一个
            if i_index == the_letter_index:
                # 输入F10 当前查找到A10 列表为[A10,B10,C10,D10]
                if i_ascii < the_letter_ascii:
                    continue
                if i_ascii > the_letter_ascii:  # 输入F10 当前查找为G10 那么应该插入在G前面
                    self.letter_queue.insert(list_i, letter)  # 当前索引为i_index 插入会把当前项往后推移
                    return
            if i_index > the_letter_index:  # 当前输入A10 列表为[A12,A13,A14...]
                self.letter_queue.insert(list_i, letter)  # 此时插入在当前项(之前)就可以
                return
        # 如果能到达此处 输入A100 当前查找到A51 列表为[A50,A51,...,A99]应该加入到末尾
        self.letter_queue.append(letter)


    def separate_letter(self, letter):
        """
        将输入拆分为两部分:
        第一个字符的ASCII值,数字的int值
        """
        the_letter_index = letter[1:]  # 从'a100'中切片得到index'100' 如果'a' 返回的是''
        the_letter_ascii = ord(letter[0])
        if the_letter_index != '':
            the_letter_index = int(the_letter_index)
        else:
            the_letter_index = 0
        return the_letter_ascii, the_letter_index

    # =========================================

class Screen_draw:
    def __init__(self, py5):
        self.py5 = py5  # Store the py5 object
        self.tools = Tools2D()

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
        screen_info = self.screen_get_info()
        x_range, y_range = screen_info['x_range'], screen_info['y_range']
        if 'directed' not in line_detail_dict.keys() or not line_detail_dict['directed']:
            raise ValueError(f"输入的有向直线有误{line_detail_dict}")
        lx, ly = line_detail_dict['location_point']
        vx, vy = line_detail_dict['direction_vector']

        print(line_detail_dict)
        line_detail = self.tools.directed_line_to_line(line_detail_dict, temp=True)
        print("x" * 10)
        print(line_detail)
        print("x" * 10)
        segment_line = self.tools.line_to_Segmentline(line_detail, x_range=x_range, y_range=y_range)
        print("!" * 10)
        print(segment_line)
        print("!" * 10)
        if not segment_line:
            return False

        segment_line_locations = self.tools.Segmentline_get_info(segment_line)['location']
        print("@" * 10)
        print(segment_line_locations)
        print("@" * 10)
        self.tools.Segmentline_remove_by_chain(segment_line)

        A_point, B_point = segment_line_locations
        Ax, Ay = A_point
        Bx, By = B_point

        vec_to_A = np.array([Ax - lx, Ay - ly])
        direction_vec = np.array([vx, vy])
        dot_product_A = np.dot(vec_to_A, direction_vec)

        if dot_product_A > 0:
            positive_point, negative_point = A_point, B_point
        else:
            positive_point, negative_point = B_point, A_point

        self.py5.stroke(color)
        self.py5.stroke_weight(stroke_weight + 4)
        self.py5.point(*line_detail_dict['location_point'])

        positive_segment_line = self.tools.Segmentline_drop(line_detail_dict['location_point'], positive_point,
                                                            **input_value)
        positive_segment_line_dict = self.tools.Segmentline_get_info(positive_segment_line)
        color_trans_segment_line_dicts = self._color_transition_segment_line(positive_segment_line_dict, **input_value,
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
