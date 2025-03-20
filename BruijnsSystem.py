# -*- coding: utf-8 -*-
import random
import sys
import time
import multiprocessing
import gc
from PY5_2DToolkit import Tools2D
import numpy as np
import pandas as pd
import warnings
from itertools import islice, product
from tabulate import tabulate
# import humanize
from joblib import Parallel, delayed

tools = Tools2D()

class BruijnsSystem:
    def __init__(self, sides: int = 5, origin_norm: int|float = 80, shifted_distance: int|float = 0, gap: int | list | tuple = 100,
                 center: tuple = (100, 100), max_num_of_line: int = 50):
        self._grid_config = {
            'sides': sides,
            'origin_norm': origin_norm,
            'shifted_distance': shifted_distance,
            'gap': gap,
            'center': center,
            'max_num_of_line': max_num_of_line
        }
        self.data_df = pd.DataFrame()
        self._inter_df = pd.DataFrame()
        self.tilling_df = pd.DataFrame()
        self._create_gird(**self._grid_config)
        self.tilling_object = None

    def __call__(self, *, sides: int = None, origin_norm: int = None, shifted_distance: int = None,
                 gap: int | list | tuple = None, center: tuple[int, int] = None, max_num_of_line: int = None):
        """
        调用实例时可以仅传入需要修改的部分参数，其余参数保持上一次的配置。
        """
        if sides is not None:
            self._grid_config['sides'] = sides
        if origin_norm is not None:
            self._grid_config['origin_norm'] = origin_norm
        if shifted_distance is not None:
            self._grid_config['shifted_distance'] = shifted_distance
        if gap is not None:
            self._grid_config['gap'] = gap
        if center is not None:
            self._grid_config['center'] = center
        if max_num_of_line is not None:
            self._grid_config['max_num_of_line'] = max_num_of_line
        self._create_gird(**self._grid_config)
        self.tilling_object = None

    @property
    def lines_dict_data(self) -> list:
        tittle = self.data_df.columns
        now_num_list = tittle[tittle.get_loc(0):]
        return self.data_df.loc[:, now_num_list].values.tolist()

    @property
    def map_df(self):
        """
        根据输入数据data_df中的origin_vector列，计算并生成一个DataFrame。
        DataFrame: 'origin_id','mirror_id','vector'。

        :attributes:
            walking (list): 存储计算后的向量映射关系，包括时钟编号、原始编号、镜像编号和向量。

        调用示例:
        调用示例,查找6的mirror_index:
        self.map_df.loc[6].index.get_level_values('mirror_id')[0]
        查找mirror_id=5 的 index:
        self.map_df.xs(5, level='mirror_id').index.get_level_values('origin_id')[0]
        """
        # xs全称是"cross-section"，用于从DataFrame中提取特定的横截面数据。
        # 它通常用于多层索引（MultiIndex），允许通过指定某个索引级别和值来快速选择数据，无需手动拆分索引。
        _map_df = pd.DataFrame()
        the_origin_vectors = self.data_df[['origin_vector','directed_vector']].T.to_dict('list')
        sides = len(the_origin_vectors)

        if sides % 2 != 0:  # 奇数
            info_zip: list = []  # clock_id,origin_id,mirror_id,vector,distance_vector
            for index, (the_vector,directed_vector) in the_origin_vectors.items():

                # 例：sides = 5
                # 圆被分成5个等分，并且存在另外5个镜像等分，总共10个分区
                # 这种排列是由于180°旋转对称导致的
                # 计算：360度/10 = 36度 -> 180度/36度 = 5
                # 索引镜像：列表长度为10，原索引 i 镜像到 (i + 镜像间隔:sides) % 总等分数:sides * 2
                # 例如: 0, 2, 4, 6, 8 分别镜像到: 5, 7, 9, 1, 3
                # 值: 1, 2, 3, 4, 5 分别变为: -1, -2, -3, -4, -5

                vx,vy = the_vector
                info_zip.append([index * 2,      #clock_id
                                 index,          #origin_id
                                 index + sides,  #mirror_id
                                 the_vector,     #origin_vector
                                 directed_vector #direction_vector
                                 ])
                # d_x,d_y = directed_vector
                info_zip.append([(index * 2 + sides) % (sides * 2),
                                 index + sides,
                                 index,
                                 [-vx,-vy],
                                 np.NAN # [-d_x,-d_y] #TODO 这里可能出错误
                                 ])

            _map_df = pd.DataFrame(info_zip, columns=['clock_id', 'origin_id', 'mirror_id', 'vector','directed_vector'])
            _map_df.set_index(['clock_id', 'origin_id', 'mirror_id'], inplace=True)
            _map_df.sort_index(level='clock_id', inplace=True)
            _map_df.index = _map_df.index.droplevel('clock_id')
        else:

            # 例: sides = 6 时，向量 0 和 3, 1 和 4, 2 和 5 互为正对
            # 为了避免在初始状态 (shift_distance = 0) 下，walking 和 walking_mirror 选取到重复的向量
            # walking_mirror 的选取需要排除 walking 中已有的向量，并选择 "对位" 的向量
            # "对位" 的向量通过 (i + 镜像间隔:sides//2) % 总等分数:sides 计算得到，它指向与 index 索引向量正对的位置

            info_zip = [
                         [index,                          #index
                         (index + sides // 2) % sides,   #mirror_index
                         the_vector,                     #origin_vector
                         d_v]                            #direction_vector
                         for index, (the_vector,d_v) in the_origin_vectors.items()
                        ]

            _map_df = pd.DataFrame(info_zip, columns=['origin_id', 'mirror_id', 'vector','directed_vector'])
            _map_df.set_index(['origin_id', 'mirror_id'], inplace=True)

        return _map_df

    @property
    def interaction_df(self):
        """
        查找所有焦点
            此方法用于查找并计算不同 'gird' （网格线组）之间所有线段的交点。
            它遍历数据 DataFrame 中表示不同 'gird' 的列，并两两比较 'gird' 内的线段，
            使用 `intersection_2line` 方法计算线段交点。
            所有找到的交点将被存储在临时字典中，并最终转换为 DataFrame 格式。
        参数:
            rebuild:是否重新完整重建inter_df,可以传入create_gird的返回值
        self.inter_df:
            包含线段交点信息的 DataFrame.列和索引均为多级索引.
            第一级索引表示 'gird' 的索引，第二级索引表示该层gird中有向直线的编号(line_num)。
            值表示交点坐标[float,float]。可以通过.loc[(index,num),(index,num)]查询任意两条线的交点
        """
        i_t = time.time()

        df_col = self.data_df.columns
        d_col = df_col[df_col.get_loc(0):]  # line的id列表
        t_i = time.time()

        print('data_loading')
        # 生成当前所有线段标识符（index, num）
        col = list(product(range(len(self.data_df)), d_col))
        # 获取对应的线段数据
        lines_np = self.data_df.loc[:, d_col].to_numpy().flatten()
        lines = lines_np.tolist()
        print('loading_use:',time.time()-t_i)

        print('start_interaction')
        #TODO 这里实际上对角线是相同的,这里重复计算了
        if self._inter_df.empty:
            inter_inf = np.around(tools.inter_line_group_np(lines_a=lines, lines_b=lines), 10)
            self._inter_df = pd.DataFrame(inter_inf.tolist(), columns=col, index=col)
        else:
            last_col = self._inter_df.columns.tolist()
            last_num = len(last_col)
            now_num = len(col)
            if now_num > last_num:
                last_col_np = np.array(last_col)
                col_np = np.array(col)

                last_col_np_mat = last_col_np[:, np.newaxis, :]  # nx1x2
                col_np_mat = col_np[np.newaxis, :, :]  # 1xmx2

                col_mask = (last_col_np_mat == col_np_mat).all(axis=2).any(axis=0)  # nxmx2 得到的是col_np_mat的mask
                index = np.where(~col_mask)[0]

                new_col = [c for t, c in enumerate(col) if t in index]  # 因为列名必须是数组形式,只能这么转换回来.
                lines_new = lines_np[index].tolist()

                inter_inf = np.around(
                    tools.inter_line_group_np(lines_a=lines, lines_b=lines_new), 10
                )
                new_df = pd.DataFrame(inter_inf.tolist(), index=col, columns=new_col)

                # 合并 new_df 与 new_df.T，生成一个包含两边信息的 DataFrame
                symmetric_new_df = new_df.combine_first(new_df.T)

                # 只用一次 combine_first 更新 self.inter_df
                self._inter_df = self._inter_df.combine_first(symmetric_new_df)

            else:
                tar_i = set(last_col) - set(col)
                tar_i = list(tar_i)
                self._inter_df.drop(index=tar_i, columns=tar_i, inplace=True)

        print('interaction_used:',time.time()-t_i)
        return self._inter_df

    @property
    def tilling(self):
        print('BS:start_load_tilling')
        if self.tilling_object:
            return self.tilling_object
        return Tilling_Create(map_df=self.map_df, inter_df=self.interaction_df)

    @staticmethod
    def _create_origin_vector_numpy(sides, radius=10):
        """
        side:边的数量(顶点的数量)
        radius:几何对象的半径
        """
        if sides < 3:
            raise ValueError("sides小于3,无法生成")
        angles_group = np.linspace(0, stop=2 * np.pi, num=sides, endpoint=False)
        # np.linspace 用于创建等间距的数值序列。 "linear space"（线性空间）
        x = np.cos(angles_group) * radius
        y = np.sin(angles_group) * radius
        back_nparray = np.column_stack([x, y])
        # all the input arrays must have same number of dimensions维度
        return Tools2D.reduce_errors_np(back_nparray)

    def _create_gird(self, sides=5, origin_norm=80, shifted_distance=0, gap: int | list | tuple = 100,
                     center=(100, 100),
                     max_num_of_line=50):
        """
        此函数用于创建一个指定参数的网格系统。

        参数:
        - sides: 网格的边数 (默认值为 5)。
        - origin_norm: 初始向量的模 (长度) (默认值为 80)。
        - shifted_distance: 向量 初始的平移距离 (默认值为 0)。
        - gap: gird内部的间距。可以是list,tuple指定每个girds的间距,int默认间距相等
        - center: 网格中心的坐标 (默认值为 (100, 100))。
        - max_num_of_line: 网格中的最大线条数 (默认值为 50)。
        返回bool:
           是否发生了项目改变,如果改变,需要重新完整的计算interaction
        """

        if not isinstance(gap, (list, tuple)):
            gap = [gap] * sides
        elif len(gap) != sides:
            raise ValueError(f"gap输入错误:{gap}")

        # 创建一组origin_vectors
        vectors_origin = self._create_origin_vector_numpy(sides, origin_norm)
        # 取vector的垂直向量vector_pen
        vectors_origin_pen = tools.vector_group_rotate_np(vectors_origin, 90).tolist()
        vectors_origin = vectors_origin.tolist()

        # 定义有向直线origin_directed_line:0
        tools.reset()
        for d_v in vectors_origin_pen:
            tools.directed_line_drop(location_point=center, direction_vector=d_v)
        line_key_list = list(tools.line_dic.keys())
        for index, line_id in enumerate(line_key_list):  # 根据distance平移
            # TODO 此处模长可以不用以相同数值平移,可以存在长度差,应该再增加一个参数调整长度差
            distance_shift_vector = tools.vector_change_norm(vectors_origin[index],
                                                             shifted_distance)
            tools.line_shift(line_id, distance_shift_vector, rewrite=True, drop=False)
        origin_directed_lines_list = list(tools.line_dic.values())

        # 避免重新完整生成
        tittle = self.data_df.columns
        key_is_in = all(a_key in tittle for a_key in ['gap', 0])
        end_num = (max_num_of_line - sides) // (2 * sides) + 1
        if key_is_in and origin_directed_lines_list == self.data_df.loc[:, 0].tolist() and gap == self.data_df[
            'gap'].tolist():
            if vectors_origin != self.data_df['origin_vector'].tolist():
                self.data_df['origin_vector'] = vectors_origin  # norm发生变化,会导致这种情况
            now_num_list = tittle[tittle.get_loc(0):]
            now_num = max(now_num_list)
            if now_num >= end_num:
                del_tar = [num for num in now_num_list if abs(num) > end_num - 1]
                if 0 in del_tar: del_tar.remove(0)  # TODO 这里有点蠢
                # print(f'当前项目减少,删除多余的line_num:{del_tar}')
                self.data_df.drop(columns=del_tar, inplace=True)
                start_num = end_num
            else:
                start_num = now_num + 1
                # print(f'当前已创建:{now_num_list} start_num:{start_num}')
        else:
            self._inter_df = pd.DataFrame()  # 此时需要重置inter
            self.data_df = pd.DataFrame()
            self.data_df['origin_vector'] = vectors_origin
            self.data_df['gap'] = gap
            self.data_df['directed_vector'] = vectors_origin_pen
            self.data_df.loc[:, 0] = origin_directed_lines_list  # 创建origin_d_line
            start_num = 1

        # 生成一个shift倍数的列表,准备遍历
        list_positive = np.arange(start_num, end_num)
        target_list = list_positive.tolist() + (-list_positive).tolist()
        self.data_df = self.data_df.reindex(columns=list(self.data_df.columns) + target_list, fill_value={})  # noqa
        # =============================== main ===============================
        # 平移gird_0，构建平行网格gird
        for index, line_dict in self.data_df.loc[:, 0].items():  # 遍历原始gird每一条线
            for i in target_list:
                o_v = self.data_df['origin_vector'][index]
                shift_distance = gap[index] * i
                shift_vector = tools.vector_change_norm(o_v, shift_distance)
                line_detail = tools.line_shift(line_dict, shift_vector, rewrite=False, drop=False)
                self.data_df.at[index, i] = line_detail  # 命名方式1,2,3...
        # =============================== main ===============================

class Tilling_Create:
    def __init__(self, map_df, inter_df):
        self.inter_df = inter_df
        self.map_df = map_df
        self.sorted_tilling_df,self.sorted_location_df = self._sorted_df()
        self.tilling_map_p = pd.DataFrame()

    def get_trend_map(self) -> dict:
        """
        根据direction_vector来确定直线走向。
        定义如下：
        - +x, +y（x递增）
        - -x, +y（x递减）
        - -x, -y（x递减）
        - +x, -y（x递增）
        该函数返回一个包含向量 x 和 y 分量符号的元组，用于指示向量在其象限中的方向。

        参数:
        line_tuple (tuple): 包含线条索引的元组。

        返回:
        tuple: 一个包含方向向量 x 和 y 分量符号（sx, sy）的元组。
        """
        tar = self.map_df['directed_vector']
        r_d = {}
        for i_,d_v in tar.items():
            if not isinstance(d_v,list):
                print('directed_vector is None')
                continue
            s_x ,s_y = np.sign(d_v[0]), np.sign(d_v[1])
            if s_x < 0 or (s_x == 0 and s_y < 0):
                r_d[i_[0]] = False
                continue
            r_d[i_[0]] = True
        return r_d
    @staticmethod
    def sort(input_df, trend_map:dict):
        the_dict = {}
        the_dict_location={}
        df_index = input_df.index
        for line_id, inter_list in input_df.items():

            arr = np.array(inter_list.tolist())

            queue_p, indices, counts = np.unique(arr, axis=0, return_index=True, return_counts=True)
            # queue_p: 排序后的队列 (去重, 默认以 [x,y] 中的 x 排序, 如果相同, 以 y 排序)
            # indices: queue_p 中元素在原数组 arr 中的序号
            # counts: 每个元素在 arr 中出现的次数

            valid_mask = np.where(~np.isnan(queue_p).any(axis=1))# 获取没有nan的序号

            # 同时过滤三个数组
            queue_p = queue_p[valid_mask]  # 过滤后的唯一值坐标
            indices = indices[valid_mask]  # 过滤后的首次出现索引
            counts = counts[valid_mask]  # 过滤后的计数 (形状与queue_p一致)

            same_v = queue_p[counts > 1]  # 取具有重复的点[x,y]
            same_id = indices[counts > 1]  # 重复点的index

            if not trend_map[line_id[0]]:
                # 需要倒序
                indices = indices[::-1]

            # [same_index: line值的index序号] 将来要把这个index值全部替换成line的index序号
            same_id_map = {
                s_id: np.where((arr == s_v).all(axis=1))[0].tolist() for s_id, s_v in zip(same_id, same_v)
            }
            indices = indices.tolist()

            for t_, i in enumerate(indices):
                if i in same_id_map:
                    indices[t_] = same_id_map.pop(i)
                    continue
                indices[t_] = [i]

            r = (
                [[df_index[_id] for _id in i_list] for i_list in indices]
                + [np.NAN] * (len(df_index) - len(indices)) #保持维度一致,不然没法添加到dataframe
                )

            r_location = queue_p.tolist() + [np.NAN] * (len(df_index) - len(queue_p))

            the_dict[line_id] = r
            the_dict_location[line_id] = r_location

        return the_dict,the_dict_location

    def _sorted_df(self):
        """
        Bruijns 系统中基于交点建立网格线邻接关系的关键预处理步骤。
        `self.inter_sorted_df` 中排序和组织的交点数据支持后续依赖网格网络有序遍历的算法，如路径查找、图构建或网格连通性特征提取。

        排序逻辑:
            - 主排序键: 交点坐标 (x, y)。利用 `numpy.unique` 默认排序，按 x 坐标升序，x 相等时按 y 升序。
            - 方向调整: 对于 x 方向分量为负，或 x 分量为零且 y 分量为负的线段，排序顺序反转，确保与预期遍历方向一致。

        输出数据结构 (`self.inter_sorted_df`):
            一个 pandas DataFrame，其中:
                - 列: 网格线标识符（元组）。
                - 值: 按线段方向排序的交点信息列表，每项为以下之一:
                    - `None`: 占位符，用于保持 DataFrame 形状。
                    - `[line_index]`: 单条线在此点相交。
                    - `[line_index_1, line_index_2, ...]`: 多条线在此点相交。
        """
        d_map = self.get_trend_map()

        inter_num = len(self.inter_df)
        num = inter_num//500 if inter_num >1000 else 1
        cut_t = time.time()

        if num >1:
            # df_chunks = np.array_split(self.inter_df, num, axis=1) #纵向切割
            chunk_size = int(np.ceil(self.inter_df.shape[1] / num))
            df_chunks = [self.inter_df.iloc[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num)]
            print('dataframe-cut:',time.time()-cut_t,'multi-process:',num)
        else:
            df_chunks = [self.inter_df]

        # 使用 joblib.Parallel 并行调用 sort 方法
        results = Parallel(n_jobs=num)(
            delayed(self.sort)(chunk, d_map) for chunk in df_chunks
        )  # backend="threading"
        print('sorted-finish:',time.time()-cut_t)

        # 合并各个分块返回的字典结果
        walk_dict = {}
        location_dict = {}
        for _dict,l_dict in results:
            walk_dict = walk_dict | _dict
            location_dict = location_dict | l_dict

        return pd.DataFrame(walk_dict).dropna(how='all'),pd.DataFrame(location_dict).dropna(how='all')

    def _get_center(self, c_point: list | tuple):
        ln = len(self.sorted_location_df)
        inter_list = self.sorted_location_df.dropna().to_numpy().tolist()
        inter_list = [i+(ln-len(i))*[np.inf,np.inf] for i in inter_list]
        inter_list = np.array(inter_list)
        distances = np.linalg.norm(inter_list - np.array(c_point), axis=-1)
        flat_min_index = np.nanargmin(distances)
        # 将展平索引转换为 (行, 列) 的二维索引
        min_row_idx, min_col_idx = np.unravel_index(flat_min_index, distances.shape)
        return min_col_idx,self.sorted_location_df.columns[min_col_idx]

    def _get_tilling_shape(self, vectors_id_list):
        """
        Get the shape of a single tiling pattern.

        这个函数计算并返回一个字典,将vector IDs映射到对应的坐标点,这些坐标点按顺时针顺序形成一个闭合多边形。

        Args:
            vectors_id_list: 一个id列表

        Returns:
            dict: 一个字典,将vector IDs 映射到segment。
                  坐标点是通过vector的累加和计算得到的,形成一个闭合多边形。
                  格式: {vector_id: segment。}
                  其中segment。是一个[x, y]坐标的列表
        """
        enable_map = self.map_df[(self.map_df.index.get_level_values('origin_id').isin(vectors_id_list)) |
                                 (self.map_df.index.get_level_values('mirror_id').isin(vectors_id_list))] #一个布尔or操作,找到所有符合要求的信息
        enable_vectors = enable_map['vector'].tolist()
        enable_id = enable_map.index.get_level_values('origin_id').tolist()
        # enable_vectors只是移动的路径,需要绘制成坐标点
        cumulative_sum = np.cumsum(np.array(enable_vectors), axis=0)
        tilling = Tools2D.reduce_errors_np(cumulative_sum, max_value=False).tolist()  # 防止出现无穷小数
        return {enable_id[_t]: [tilling[_t-1],tilling[_t]] for _t in range(len(tilling))}

    @staticmethod
    def _line_tups_to_vectors_id (line_tups):
        #line_tup: (o_v,line_num))
        return [i[0] for i in line_tups]

    def _loc_to_tilling_shape(self, loc):
        index, id_tup = loc
        inter_info = self.sorted_tilling_df.loc[index, id_tup] + [id_tup] #得带上id_tup自己 #TODO 会超出范围
        vectors_list = self._line_tups_to_vectors_id(inter_info)
        return self._get_tilling_shape(vectors_list)

    def _next_loc_list(self, loc)->tuple[list[tuple],list[tuple]]:
        """
        获取当前坐标位置的所有相关位置及其方向。

        参数:
        loc (list): 当前坐标的位置。形式为 [index, (origin_id, line_num)]，其中:
            - index: 在 `sorted_df` 中的 index
            - (origin_id, line_num): 在 `sorted_df` 中的 column

        返回:
        tuple: 包含两个元素:
            1. now_loc (list): 当前坐标和所有等价坐标的列表，每个坐标为 [index, (row, col)] 的形式。
            2. next_positions (list): 与等价坐标相关的方向信息，格式为 [[loc, direction], [loc, direction], ...]，
               其中 direction 将来拼接时的接口数据。
        """
        index,id_tup = loc
        now_loc = [ tuple(loc) ]
        next_:list[tuple] = [ ((index + 1, id_tup), id_tup[0]), ((index - 1, id_tup), self.origin_to_mirror(id_tup[0])) ]
        equal_loc: list[tuple] = self.loc_map(loc)

        for loc in equal_loc:
            s_id,colum = loc
            #这里要判断s_id+1 和 s_id-1 是否是nan,如果是应该跳过.
            direction_posit,_ = colum #(o_vector,line_num)
            next_.append(((s_id+1,colum),direction_posit ))
            next_.append(( (s_id-1,colum),self.origin_to_mirror(direction_posit) ))

        return equal_loc + now_loc, next_  # [ [[loc],direction],[[loc],direction],[[loc],..],... ]

    def origin_to_mirror (self, o_id):
        return self.map_df.xs(o_id,level='origin_id').index.get_level_values('mirror_id').tolist()[0]

    def mirror_to_origin (self,m_id):
        return self.map_df.xs(m_id, level='mirror_id').index.get_level_values('origin_id').tolist()[0]

    def unique_tilling(self,loc,origin_shape=None)->tuple[list[tuple],dict]:
        #loc: (index,(o_v,num))
        if origin_shape is None:
            origin_shape = self._loc_to_tilling_shape(loc)
        now_loc,next_data = self._next_loc_list(loc)
        r_dict = {tuple(loc):origin_shape}
        for next_loc,direction in next_data:
            back = self.splice_tilling(origin_shape,self._loc_to_tilling_shape(next_loc),direction)
            r_dict[tuple(next_loc)] = back
        return now_loc,r_dict

    def loc_map(self,loc)->list[tuple]|None:
        """
        查找与给定坐标等价的其他位置。

        参数:
        loc (list): 当前坐标。形式为 [index, (o_v_id, line_num)]，其中:
            - index: 在 `sorted_df` 中的索引。
            - (o_v_id, line_num): 一个包含两个整数的元组，表示该位置的 列索引

        返回:
        list | None: 返回一个列表，其中包含等价位置的索引和列的元组。格式为 [[index, (int, int)], ...]。
                    如果没有找到等价点，则返回 `None`。

        例：loc = [2, (1, 3)]
        return：[[3, (2, 3)], [4, (3, 3)]]
        """
        r=[]
        _index,line_tup = loc
        other_col = self.sorted_tilling_df.loc[_index,line_tup]
        # print(other_col)
        for col in other_col:
            bool_list = self.sorted_tilling_df.loc[:, col].dropna().apply(lambda x: line_tup in x).to_numpy()
            if np.all(bool_list==False):
                 continue
            _index = np.where(bool_list)[0][0] #查找位置
            r.append((_index,col))
        if r:
            return r
        warnings.warn('没有找到等价点')
        return None

    def WFS(self, deep:int=100):
        loc = self._get_center([100, 100]) #找到起点
        print('=='*15)
        print(f'开始:{loc}')
        now_loc,data = self.unique_tilling(loc) #(当前点的所有名称,起点形状)

        e_ = set()
        for i in data.keys():
            e_ = e_ | {e for e in self.loc_map(i)} | {i}
        done = {i for i in now_loc} | e_
        print(f'1st,done:{done}')
        # 接下来的起点
        loc_queue = list(data.keys())
        loc_queue.remove(loc)  # 上次的起点不需要,删除掉

        while loc_queue:
            if deep == 0: break
            next_loc = loc_queue.pop(0) #下一个需要遍历的起点
            print('当前loc:',next_loc)
            now_loc,r = self.unique_tilling(next_loc, data[next_loc]) #获取 (所有名称,新形状)

            del_tar = {i for i in r.keys()} & done
            for i in del_tar:
                print(f'del,{i}')
                r.pop(i)
            if not r:
                print('all_del')
                continue
            print(f"剩余:{r.keys()}")
            data.update(r)  # 更新data

            e_ = set()#当前创建的所有点
            for i in r.keys():
                e_= e_|{e for e in self.loc_map(i)}|{i}
            done = done | {i for i in now_loc} | e_
            print(f'done:{done}')

            loc_queue = loc_queue + list(r.keys())  # 加入到队列
            deep -= 1

        seg_data = []
        print('WFS-search-finish')
        for i in list(data.values()):
            seg = list(i.values())
            seg_data.extend(seg)
        return seg_data

    def splice_tilling(self,a_tilling,b_tilling,direction=None):
        #direction: origin_vector 代表a与b重合的位置(从a的角度),例:a的1和b的6重合-->direction=1

        # #临时的方法
        # if not direction:
        #     a_v = a_tilling.keys()
        #     a_v_mirror = [self.origin_to_mirror(_v) for _v in a_v]
        #     for a_v_m in a_v_mirror:
        #         if a_v_m in b_tilling:
        #             direction = self.mirror_to_origin(a_v_m)
        #     print('direction: ',direction)
        #     print('a_tilling:',a_tilling)
        #     print('b_t...:',b_tilling)
        #     if not direction:
        #         raise ValueError('not direction 还是没找到!')

        #================main===============

        # 因为两次直线方向相反,所以第一个取[0],第二个取[1]

        b_same = b_tilling[self.mirror_to_origin(direction)][0]

        o_same = a_tilling[direction][1]

        shift_v = [o_same[0] - b_same[0], o_same[1] - b_same[1]]

        b_tilling = {k:tools.point_shift(v,shift_v)for k,v in b_tilling.items()}

        return b_tilling









def pd_print(df: pd.DataFrame, max_length=20, multi_index=False):
    """
    打印整个DataFrame，不论其大小，长值会被从中间缩略显示。

    :param df: 需要打印的 DataFrame
    :param max_length: 字符串的最大显示长度
    :param multi_index: 是否显示多级索引
    """

    def truncate_middle(val):
        # def format_float(n):
        #     if isinstance(n, float):
        #         rounded = round(n, 2)
        #         # 去掉末尾的无用零和小数点
        #         str_val = f"{rounded:.2f}".rstrip('0').rstrip('.')
        #         # 如果原数值在最后一个有效位数之后还有更多小数，加上'..'
        #         if len(f"{n:.15f}".split('.')[-1].rstrip('0')) > 2:
        #             return f"{str_val}.."
        #         return str_val
        #     return n
        #
        # if isinstance(val, (list, tuple, np.ndarray)):
        #     formatted_val = [format_float(n) for n in val]
        #     # 将所有元素转换为字符串然后拼成一个字符串，并用 [] 包起来
        #     val_str = "[" + ", ".join(map(str, formatted_val)) + "]"
        #
        #     return val_str
        val_str = str(val)
        if len(val_str) > max_length:
            half_length = (max_length - 3) // 2
            return val_str[:half_length] + '...' + val_str[-half_length:]
        return val_str

    df_shortened = df.apply(lambda col: col.map(lambda x: truncate_middle(x)))
    if multi_index:
        df_shortened.reset_index(inplace=True)
        print(tabulate(df_shortened, headers='keys', tablefmt="pretty", showindex=False))
    else:
        print(tabulate(df_shortened, headers='keys', tablefmt="pretty"))  # orgtbl #presto #pretty #github

def deep_get_size(obj, seen=None):
    """
    递归计算对象的深层内存大小 (单位: 字节)
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum([deep_get_size(v, seen) + deep_get_size(k, seen) for k, v in obj.items()])
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum([deep_get_size(item, seen) for item in obj])
    # 可以根据需要添加其他容器类型的处理，例如自定义对象
    return size


if __name__ == "__main__":
    a = BruijnsSystem(sides=5, max_num_of_line=20, shifted_distance=0)
    a(sides=5, max_num_of_line=500, shifted_distance=120,gap=12)
    t= a.tilling
    print(t.WFS(9))
    print(t.WFS(10))