import json
import logging
import math
import multiprocessing
import numpy as np
from configs.config import PLTV_LEVEL, max_search_num, max_sampling_freq, Multi_Process, Environment, No_pltv, MAB_SAVE_STEP
import copy
from collections import defaultdict
from bandit_public.calculateDelta import CalculateDelta
import os
from datetime import datetime
import pandas as pd
from registry import registry


if Environment == "offline":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d ]in %(funcName)-8s  %(message)s"
    )
else:
    logging.basicConfig(
        filename=os.path.join(os.getcwd(),
                              "./log/" + __file__.split(".")[0] + datetime.datetime.strftime(datetime.datetime.today(),
                                                                                             "_%Y%m%d") + ".log"),
        level=logging.INFO,
        filemode="a+",
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d ]in %(funcName)-8s  %(message)s"
    )


@registry.register('UCBBanditNoPrior')
class UCBBandit(object):
    """
    bid shading 主类
    """

    def __init__(self):
        """
        初始化
        """
        self.calculate_delta = CalculateDelta()

    def calculate_market_price(self, media_app_id, position_id, market_price_dict,
                               impression_price_dict, no_impression_price, norm_dict, optimal_ratio_dict,
                               data_pd):
        """
        计算市场价格
        :params market_price_dict = {pltv:value}
        :params impression_price_dict = {pltv:value_list}
        :params no_impression_price = {pltv:value_list}  响应未曝光数据
        """

        # """分pltv计算"""
        if not No_pltv:
            for level in PLTV_LEVEL:
                market_price_value = -1.0
                impression_price_list = []
                no_impression_price_list = []
                if level in market_price_dict:
                    market_price_value = market_price_dict[level]

                if level in impression_price_dict:
                    impression_price_list = sorted(impression_price_dict[level], reverse=False)

                if level in no_impression_price:
                    no_impression_price_list = sorted(no_impression_price[level], reverse=False)

                if market_price_value == -1.0 or len(impression_price_list) < 100:
                    logging.debug(f"proc_id={multiprocessing.current_process().name},"
                                  f"media_app_id:{media_app_id}, position_id:{position_id}, level:{level},"
                                  f"len(impression_price_list):{len(impression_price_list)} < 10, data is sparse "
                                  f"enough to compute")
                    continue

                # e-e
                # market_price 市场价格
                # chosen_count_map = {}  # 记录选择次数
                # imp_count_map = {}  # 记录曝光次数
                market_price, chosen_count_map, imp_count_map, true_imp_count_map,\
                true_chosen_count_map, revenue_rate_list,optimal_ratio_dict = self.bandit(media_app_id,
                                                                                           position_id,
                                                                                           norm_dict[level],
                                                                                           market_price_value,
                                                                                           impression_price_list,
                                                                                           no_impression_price_list,
                                                                                           data_pd,
                                                                                           optimal_ratio_dict)

                logging.info(f"proc_id={multiprocessing.current_process().name},"
                             f"media_app_id:{media_app_id}, position_id:{position_id}, level:{level}, "
                             f"history_median_price_value:{market_price_value}, market_price:{market_price}，"
                             f"len impression_price_list:{len(impression_price_list)}, "
                             f"len no_impression_price_list:{len(no_impression_price_list)}")

                optimal_ratio_dict = self.save_bandit_result(media_app_id, position_id, level,
                                                             market_price, chosen_count_map, imp_count_map,
                                                             norm_dict, optimal_ratio_dict)
        else:
            # """计算position默认值 """
            market_price_value = round(np.mean(market_price_dict), 2)
            impression_price_list = impression_price_dict
            no_impression_price_list = no_impression_price
            impression_price_list = sorted(impression_price_list, reverse=False)
            no_impression_price_list = sorted(no_impression_price_list, reverse=False)

            if market_price_value == -1.0 or len(impression_price_list) < 100:
                logging.debug(f"proc_id={multiprocessing.current_process().name},"
                              f"media_app_id:{media_app_id}, position_id:{position_id}, level: None,"
                              f"len(impression_price_list):{len(impression_price_list)} < 10, data is sparse "
                              f"enough to compute")
            else:
                # e-e
                # market_price 市场价格
                # chosen_count_map = {}  # 记录选择次数
                # imp_count_map = {}  # 记录曝光次数
                market_price, chosen_count_map, imp_count_map, true_imp_count_map,true_chosen_count_map,\
                    revenue_rate_list, optimal_ratio_dict = self.bandit(media_app_id, position_id, norm_dict,
                                                                        market_price_value,
                                                                        impression_price_list,
                                                                        no_impression_price_list,
                                                                        data_pd,
                                                                        optimal_ratio_dict)

                logging.info(f"calculate default data proc_id={multiprocessing.current_process().name},"
                             f"media_app_id:{media_app_id}, position_id:{position_id},"
                             f"history_median_price_value:{market_price_value}, market_price:{market_price}，"
                             f"len impression_price_list:{len(impression_price_list)}, "
                             f"len no_impression_price_list:{len(no_impression_price_list)}")

                # optimal_ratio_dict = self.save_bandit_result(media_app_id, position_id, -1,
                #                                              market_price, chosen_count_map, imp_count_map,
                #                                              norm_dict, true_imp_count_map)

        return optimal_ratio_dict

    def save_bandit_result(self, media_app_id, position_id, level, market_price_norm,
                           chosen_count_map, imp_count_map, norm_dict, optimal_ratio_dict):

        norm_max = norm_dict["norm_max"]
        norm_min = norm_dict["norm_min"]
        market_price = market_price_norm * (norm_max - norm_min) + norm_min

        if level == -1:
            key = f"{media_app_id}_{position_id}"
        else:
            key = f"{media_app_id}_{position_id}_{level}"

        if key not in optimal_ratio_dict:
            optimal_ratio_dict[key] = {}

        optimal_ratio_dict[key]['market_price'] = market_price
        optimal_ratio_dict[key]['chosen_count_map'] = chosen_count_map
        optimal_ratio_dict[key]['imp_count_map'] = imp_count_map
        optimal_ratio_dict[key]['norm_dict'] = norm_dict
        # optimal_ratio_dict[key]['true_imp_count_map'] = true_imp_count_map
        # optimal_ratio_dict[key]['true_chosen_count_map'] = true_chosen_count_map
        # optimal_ratio_dict[key]['reward_ratio_list'] = reward_ratio_list

        return optimal_ratio_dict

    def save_bandit_result_during_loop(self, media_app_id, position_id, level,
                                       market_price_norm, chosen_count_map, imp_count_map, true_chosen_count_map,
                                       true_imp_count_map, norm_dict, loop_index, optimal_ratio_dict):

        norm_max = norm_dict["norm_max"]
        norm_min = norm_dict["norm_min"]
        market_price = market_price_norm * (norm_max - norm_min) + norm_min

        if level == -1:
            key = f"{media_app_id}_{position_id}"
        else:
            key = f"{media_app_id}_{position_id}_{level}"

        if key not in optimal_ratio_dict:
            optimal_ratio_dict[key] = {}

        if loop_index not in optimal_ratio_dict[key]:
            optimal_ratio_dict[key][loop_index] = {}

        optimal_ratio_dict[key][loop_index]['market_price'] = market_price
        optimal_ratio_dict[key][loop_index]['chosen_count_map'] = chosen_count_map
        optimal_ratio_dict[key][loop_index]['imp_count_map'] = imp_count_map
        optimal_ratio_dict[key][loop_index]['norm_dict'] = norm_dict
        optimal_ratio_dict[key]['true_imp_count_map'] = true_imp_count_map
        optimal_ratio_dict[key]['true_chosen_count_map'] = true_chosen_count_map
        # optimal_ratio_dict[key]['reward_ratio_list'] = reward_ratio_list

        return optimal_ratio_dict

    def calculate_reward_weigth(self, price, market_price_value, right_range, left_range):
        """
        计算reward权重
        """
        reward = 0.01
        if market_price_value * 0.9 < price <= market_price_value * 1.1:
            reward = 0.9
        elif price > market_price_value * 1.1:
            reward = 1 - ((price - market_price_value) / right_range)
            if reward > 0.9:
                reward = 0.9
        else:
            reward = 1 - ((market_price_value - price) / left_range)
            if reward > 0.9:
                reward = 0.9

        return reward

    # todo(mfishznag) 实验2 weight形式
    def calculate_reward_weigt_quadratic(self, price, market_price_value):
        """
        计算reward权重
        """
        # reward = 1 - (price - market_price_value) ** 2
        # 线性
        reward = 1 / np.exp(np.abs(price - market_price_value))
        # reward = 1
        return reward

    def bandit_init(self, impression_price_list, no_impression_price_list, market_price_value):
        """
        # bandit 初始化
        """
        estimared_rewards_map = {}  # 记录reward
        chosen_count_map = {}  # 记录选择次数
        imp_count_map = {}  # 记录曝光次数

        # right_range = max(abs(impression_price_list[-1] - market_price_value), 1)
        # left_range = max(abs(market_price_value - impression_price_list[0]), 1)

        # 步骤1：初始化
        for price in impression_price_list:  # 曝光数据
            if price not in chosen_count_map:
                chosen_count_map[price] = 1
            else:
                chosen_count_map[price] += 1

            if price not in imp_count_map:  # 响应有曝光
                imp_count_map[price] = 1
            else:
                imp_count_map[price] += 1

        for price in no_impression_price_list:  # 响应未曝光
            # price_set.add(price)
            if price not in chosen_count_map:
                chosen_count_map[price] = 1
            else:
                chosen_count_map[price] += 1

        for price in chosen_count_map.keys():
            rate = 0.0
            if price in imp_count_map:
                rate = float(imp_count_map[price]) / chosen_count_map[price]

            estimared_rewards_map[price] = rate * self.calculate_reward_weigt_quadratic(price, market_price_value)
        return chosen_count_map, imp_count_map, estimared_rewards_map

    def bandit(self, media_app_id, position_id, norm_dict, market_price_value,
               impression_price_list, no_impression_price_list, data_pd, optimal_ratio_dict):
        """
        e-e探索：UCB方式
        """
        # 二价数据集未竞得的win price=0，如果使用回传市场价格的数据，这里需要修改
        data_pd = data_pd[data_pd.win_price <= data_pd.response_ecpm]

        # 步骤1：初始化
        chosen_count_map, imp_count_map, estimared_rewards_map = self.bandit_init(impression_price_list,
                                                                                  no_impression_price_list,
                                                                                  market_price_value)
        for k in list(chosen_count_map.keys()):
            imp_count_map[k] = 1
            chosen_count_map[k] = 2

        true_chosen_count_map = copy.deepcopy(chosen_count_map)
        true_imp_count_map = copy.deepcopy(imp_count_map)

        """
        for key, _ in chosen_count_map.items():
            chosen_count_map[key] = 2
            imp_count_map[key] = 1
        """

        # 步骤2：选top
        chosen_key_set = list(chosen_count_map.keys())
        if len(chosen_count_map) > max_search_num:
            # 为了减少计算时间只取top max_search_num 进行计算
            chosen_count_sorted = sorted(chosen_count_map.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            chosen_key_set = set((i[0] for i in chosen_count_sorted[:max_search_num]))

        price_list = list(chosen_key_set)
        price_list = sorted(price_list, reverse=False)
        len_price_list = len(price_list)

        # 步骤3： bandit 计算
        """      
        先验 + 模拟（played）的总计数：
            chosen_count_map: played nums，用于计算alpha，beta
            imp_count_map: success nms，用于计算alpha，beta
        模拟（pull）计数：
            sampling_chosen_count_map: pull nums，用于计算UCB
        type A更新计数：
            type_a_update: k > market_price --> k is played && imp[k] += 1，作为reward的分母
        """
        sampling_chosen_count_map = {}
        revenue_rate_list = []
        type_a_update = defaultdict(int)

        search_count_set = []

        loop_index = 0
        for _, row in data_pd.iterrows():
            ecpm = row["response_ecpm"]
            win_price = row["win_price"]

            max_upper_bound_probs = 0.0
            max_probs_key = 0
            total_count = sum(sampling_chosen_count_map.values())
            # 步骤3：1、select arms
            for k in chosen_key_set:
                sampling_count = 0
                if k in sampling_chosen_count_map:
                     sampling_count += sampling_chosen_count_map[k]

                # 计算I
                if k in imp_count_map:
                    alpha = max(imp_count_map[k], 1)
                    beta = max(chosen_count_map[k] - imp_count_map[k], 1)
                else:
                    alpha = 1
                    beta = max(chosen_count_map[k], 1)

                I = alpha / (alpha + beta) + (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

                upper_bound_probs = estimared_rewards_map[k] / (type_a_update[k] + 1) * I \
                                    + self.calculate_delta.sqrt(total_count, sampling_count)
                if max_upper_bound_probs < upper_bound_probs:
                    max_upper_bound_probs = upper_bound_probs
                    max_probs_key = k

            # 记录上一轮的reward ratio
            revenue_rate_list.append(max_upper_bound_probs)

            if max_probs_key == 0:
                continue

            if max_probs_key not in sampling_chosen_count_map:
                sampling_chosen_count_map[max_probs_key] = 0
            sampling_chosen_count_map[max_probs_key] += 1

            # 步骤3：2、update
            if max_probs_key in imp_count_map:
                # np.random.randn(1)[0] -> 改为基于历史数据的采样
                # beta 先验  float(imp_count_map[max_probs_key]) / chosen_count_map[max_probs_key]
                sample_rate = np.random.beta(imp_count_map[max_probs_key],
                                             max(chosen_count_map[max_probs_key] - imp_count_map[max_probs_key], 1))
                is_win = np.random.binomial(1, sample_rate)
                index = price_list.index(max_probs_key)

                #####
                if win_price == 0 and max_probs_key < ecpm:
                    is_win = 0
                if win_price > 0:
                    if max_probs_key >= win_price:
                        is_win = 1
                    else:
                        is_win = 0
                #####

                count = 0
                if is_win == 1:
                    # 选择的max_probs_key能曝光，向左搜索（减价）
                    index = price_list.index(max_probs_key)
                    index -= 1
                    while index >= 0:

                        count -= 1

                        tmp_price = price_list[index]
                        if tmp_price not in imp_count_map or imp_count_map[tmp_price] < 1 \
                                or chosen_count_map[tmp_price] - imp_count_map[tmp_price] < 1:
                            break

                        sample_rate = np.random.beta(imp_count_map[tmp_price],
                                                     chosen_count_map[tmp_price] - imp_count_map[tmp_price])
                        is_win = np.random.binomial(1, sample_rate)
                        if is_win == 1:
                            # 向小于方向探索
                            index -= 1
                        else:
                            break

                    index += 1
                else:
                    # 选择的max_probs_key不能曝光，向右搜索（加价）
                    index += 1
                    while index < len_price_list:

                        count += 1

                        tmp_price = price_list[index]
                        if tmp_price not in imp_count_map or imp_count_map[tmp_price] < 1 \
                                or chosen_count_map[tmp_price] - imp_count_map[tmp_price] < 1:
                            break

                        sample_rate = np.random.beta(imp_count_map[tmp_price],
                                                     chosen_count_map[tmp_price] - imp_count_map[tmp_price])
                        is_win = np.random.binomial(1, sample_rate)
                        if is_win != 1:
                            # 向大于方向探索
                            index += 1
                        else:
                            break

                    index -= 1

                search_count_set.append(count)

                min_market_price = price_list[index]
                for x in chosen_key_set:
                    chosen_count_map[x] += 1
                    if x >= min_market_price:
                        type_a_update[x] += 1
                        if x not in imp_count_map.keys():
                            imp_count_map[x] = 0
                        imp_count_map[x] += 1  # if x == max_probs_key else 0.1

                        weight = self.calculate_reward_weigt_quadratic(x, min_market_price)
                        estimared_rewards_map[x] += 1 * weight

            loop_index += 1
            if loop_index % MAB_SAVE_STEP == 0:
                imp_count_map_now = {}
                chosen_count_map_now = {}
                for x in chosen_count_map.keys():
                    if x in true_imp_count_map:
                        imp_count_map_now[x] = imp_count_map[x] - true_imp_count_map[x]
                    chosen_count_map_now[x] = chosen_count_map[x] - true_chosen_count_map[x]

                optimal_ratio_dict = self.save_bandit_result_during_loop(media_app_id, position_id, -1,
                                                                         0, chosen_count_map_now, imp_count_map_now,
                                                                         true_chosen_count_map, true_imp_count_map,
                                                                         norm_dict, loop_index, optimal_ratio_dict)

        # 取reward最大值
        market_price = 0
        market_price_score = 0.0
        for price, value in estimared_rewards_map.items():
            if value > market_price_score:
                market_price_score = value
                market_price = price
            estimared_rewards_map[price] = value / max_sampling_freq

        for x in chosen_count_map.keys():
            if x in true_imp_count_map:
                imp_count_map[x] = imp_count_map[x] - true_imp_count_map[x]
            chosen_count_map[x] = chosen_count_map[x] - true_chosen_count_map[x]

        return market_price, chosen_count_map, imp_count_map, \
               true_imp_count_map, true_chosen_count_map, revenue_rate_list, optimal_ratio_dict

    def do_process(self, media_app_id, media_position_dict_obj, market_price_dict_obj, impression_price_dict_obj,
                   no_impression_obj, norm_dict, data_pd):
        """
        根据读取的数据，计算bid shading系数，输出至redis
        :return:
        """
        optimal_ratio_dict = {}

        if media_app_id not in media_position_dict_obj:
            return optimal_ratio_dict

        position_set = media_position_dict_obj[media_app_id]

        logging.info(f"proc_id={multiprocessing.current_process().name}, "
                     f"media_app_id:{media_app_id}, position_set:{position_set}")

        if Multi_Process:
            data_pd = pd.DataFrame.from_dict(dict(data_pd), orient='columns')

        logging.info(f"data_pd_head:{data_pd.head()}")

        for position_id in position_set:
            # self.market_price_dict = media_app_id:position_id:pltv - value
            market_price = {}
            if media_app_id in market_price_dict_obj \
                    and position_id in market_price_dict_obj[media_app_id]:
                market_price = market_price_dict_obj[media_app_id][position_id]

            # self.impression_price_dict = media_app_id:position_id:pltv - value_list
            impression_price = {}
            if media_app_id in impression_price_dict_obj \
                    and position_id in impression_price_dict_obj[media_app_id]:
                impression_price = impression_price_dict_obj[media_app_id][position_id]

            # self.impression_price_dict = media_app_id:position_id:pltv - value_list
            no_impression_price = {}
            if media_app_id in no_impression_obj \
                    and position_id in no_impression_obj[media_app_id]:
                no_impression_price = no_impression_obj[media_app_id][position_id]

            if not No_pltv:
                if len(market_price) < 1 or len(impression_price) < 1 or len(no_impression_price) < 1:
                    # 数据不合格跳过
                    logging.info(f"len(market_price):{len(market_price)} < 1 "
                                 f"or len(impression_price):{len(impression_price)} < 1 "
                                 f"or len(no_impression_price):{len(no_impression_price)} < 1")
                    continue

            optimal_ratio_dict = \
                self.calculate_market_price(media_app_id, position_id, market_price, impression_price,
                                            no_impression_price, norm_dict[position_id], optimal_ratio_dict,
                                            data_pd[data_pd.position_id == position_id])

        return optimal_ratio_dict
