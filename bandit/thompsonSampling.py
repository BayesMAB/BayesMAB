# https://www.aionlinecourse.com/tutorial/machine-learning/thompson-sampling-intuition


import json
import logging
import math
import multiprocessing
import numpy as np
from configs.config import max_search_num, max_sampling_freq, Multi_Process, Environment, MAB_SAVE_STEP
import copy
from collections import defaultdict
import pandas as pd
import os
from scipy.stats import beta as Beta
from datetime import datetime
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


@registry.register('ThompsonSamplingBandit')
class ThompsonSamplingBandit(object):
    """
    bid shading
    """

    def __init__(self):
        """

        """

    def calculate_market_price(self, media_app_id, position_id, market_price_dict, impression_price_dict,
                               no_impression_price, norm_dict, optimal_ratio_dict, data_pd):

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

        return optimal_ratio_dict

    def save_bandit_result(self, media_app_id, position_id, level,
                           market_price_norm, chosen_count_map, imp_count_map, norm_dict,
                           optimal_ratio_dict):

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

    def calculate_reward_weigt_quadratic(self, price, market_price_value):

        reward = 1 / np.exp(np.abs(price - market_price_value))

        return reward

    def bandit_init(self, impression_price_list, no_impression_price_list, market_price_value):
        estimared_rewards_map = {}
        chosen_count_map = {}
        imp_count_map = {}

        for price in impression_price_list:
            if price not in chosen_count_map:
                chosen_count_map[price] = 1
            else:
                chosen_count_map[price] += 1

            if price not in imp_count_map:
                imp_count_map[price] = 1
            else:
                imp_count_map[price] += 1

        for price in no_impression_price_list:
            if price not in chosen_count_map:
                chosen_count_map[price] = 1
            else:
                chosen_count_map[price] += 1

        ecpm_alpha = {}
        ecpm_beta = {}
        for price in chosen_count_map.keys():
            rate = 0.0
            if price in imp_count_map:
                rate = float(imp_count_map[price]) / chosen_count_map[price]

            estimared_rewards_map[price] = rate * self.calculate_reward_weigt_quadratic(price, market_price_value)

            ecpm_beta[price] = 1
            ecpm_alpha[price] = 1

        return chosen_count_map, imp_count_map, estimared_rewards_map, ecpm_alpha, ecpm_beta

    def select_arm(self, chosen_key_set, ecpm_alpha, ecpm_beta):
        """
        Thompson Sampling selection of arm for each round
        """
        max_probs_key = 0
        beta_rvs_best = 0.0
        for k in chosen_key_set:
            alpha = ecpm_alpha[k]
            beta = ecpm_beta[k]

            # Perform random draw for all arms based on their params (a,b)
            beta_rvs = Beta.rvs(alpha, beta, size=1)
            if beta_rvs_best < beta_rvs:
                beta_rvs_best = beta_rvs
                max_probs_key = k

        return max_probs_key, beta_rvs_best

    def bandit(self, media_app_id, position_id, norm_dict, market_price_value,
               impression_price_list, no_impression_price_list, data_pd, optimal_ratio_dict):
        data_pd = data_pd[data_pd.win_price <= data_pd.response_ecpm]

        chosen_count_map, imp_count_map, estimared_rewards_map,\
            ecpm_alpha, ecpm_beta = self.bandit_init(impression_price_list, no_impression_price_list,
                                                     market_price_value)

        true_chosen_count_map = copy.deepcopy(chosen_count_map)
        true_imp_count_map = copy.deepcopy(imp_count_map)

        chosen_key_set = list(chosen_count_map.keys())
        if len(chosen_count_map) > max_search_num:
            chosen_count_sorted = sorted(chosen_count_map.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            chosen_key_set = set((i[0] for i in chosen_count_sorted[:max_search_num]))

        price_list = list(chosen_key_set)
        price_list = sorted(price_list, reverse=False)
        len_price_list = len(price_list)

        sampling_chosen_count_map = {}
        revenue_rate_list = []
        type_a_update = defaultdict(int)

        search_count_set = []

        loop_index = 0

        for _, row in data_pd.iterrows():
            ecpm = row["response_ecpm"]
            win_price = row["win_price"]
            max_probs_key, beta_rvs_best = self.select_arm(chosen_key_set, ecpm_alpha, ecpm_beta)

            revenue_rate_list.append(beta_rvs_best)

            if max_probs_key == 0:
                continue

            if max_probs_key not in sampling_chosen_count_map:
                sampling_chosen_count_map[max_probs_key] = 0

            sampling_chosen_count_map[max_probs_key] += 1

            if max_probs_key in imp_count_map:
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
                            index -= 1
                        else:
                            break

                    index += 1
                else:
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
                        imp_count_map[x] += 1

                        weight = self.calculate_reward_weigt_quadratic(x, min_market_price)
                        estimared_rewards_map[x] += 1 * weight

                        # ecpm_alpha is based on total counts of rewards of arm
                        ecpm_alpha[x] += 1
                    else:
                        # ecpm_beta is based on total counts of failed rewards on arm
                        ecpm_beta[x] += 1

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
            market_price = {}
            if media_app_id in market_price_dict_obj \
                    and position_id in market_price_dict_obj[media_app_id]:
                market_price = market_price_dict_obj[media_app_id][position_id]

            impression_price = {}
            if media_app_id in impression_price_dict_obj \
                    and position_id in impression_price_dict_obj[media_app_id]:
                impression_price = impression_price_dict_obj[media_app_id][position_id]

            no_impression_price = {}
            if media_app_id in no_impression_obj \
                    and position_id in no_impression_obj[media_app_id]:
                no_impression_price = no_impression_obj[media_app_id][position_id]

            optimal_ratio_dict = \
                self.calculate_market_price(media_app_id, position_id, market_price, impression_price,
                                            no_impression_price, norm_dict[position_id], optimal_ratio_dict,
                                            data_pd[data_pd.position_id == position_id])
        return optimal_ratio_dict
