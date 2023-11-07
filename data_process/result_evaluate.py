import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from configs.config import No_pltv, MAB_SAVE_STEP, EVALUATION_POINT_NUMS, EVALUATION_POINT_STEP
from data_process.read_data import ReadData
from search.calculate_price_adjustment_gain import calculate_price_adjustment_gain
# from tools.reward_ratio_result_plot import Reward_Ratio_Image


class ResultEvaluate(object):
    """
    # 数据读取主类
    """

    def __init__(self, logging, TEST_DATA_PATH):
        """
        # 初始化
        """
        self.logging = logging
        self.data_list = []
        self.cal_price_adjustment_gain = calculate_price_adjustment_gain(self.logging)
        self.shading_result_list = []
        self.TEST_DATA_PATH = TEST_DATA_PATH

    def read_data(self):
        read_data = ReadData(logging=self.logging, data_path=self.TEST_DATA_PATH)
        self.data_list = read_data.test_data_process()

    def analyze_bandit_dict(self, bandit_dict):
        bandit_result = {}
        for key, optimal_ratio_dict in bandit_dict.items():
            key_list = key.split("_")
            media_app_id = key_list[0]
            position_id = key_list[1]
            if media_app_id not in bandit_result:
                bandit_result[media_app_id] = {}

            if No_pltv:
                bandit_result[media_app_id][position_id] = optimal_ratio_dict
            else:
                pltv_level = -1
                if not No_pltv and len(key_list) == 3:
                    pltv_level = key_list[2]

                position_dict = bandit_dict[media_app_id]
                if position_id not in position_dict:
                    position_dict[position_id] = {}

                position_dict[pltv_level] = optimal_ratio_dict
                bandit_result[media_app_id][position_id] = position_dict

        return bandit_result

    def result_evaluation(self, result):
        """
        评估search
        """
        evaluation = {}
        self.logging.info(f"result.keys():{result.keys()}\n"
                          f"test_dataset:{self.data_list.head(100)}")

        for key, result_dict in result.items():
            test_pd = self.data_list[self.data_list.key == key]
            if test_pd.empty:
                self.logging.info(f"key:{key}, test_pd is empty!")
                continue

            # self.logging.info(f"key:{key}, test_pd:\n{test_pd.head()}")
            market_price = result_dict["market_price"]
            chosen_count_map = result_dict["chosen_count_map"]
            imp_count_map = result_dict["imp_count_map"]
            norm_dict = result_dict["norm_dict"]
            norm_max = float(norm_dict["norm_max"])
            norm_min = float(norm_dict["norm_min"])
            price, opt_gain, before_gain \
                = self.cal_price_adjustment_gain.get_adjust_price(test_pd[["response_ecpm", "win_price", "click_num",
                                                                           "target_cpa", "pay_amount"]],
                                                                  market_price, chosen_count_map,
                                                                  imp_count_map, norm_dict)

            # 二价数据集测试效果
            test_pd = test_pd.copy()
            test_pd["shading_ecpm"] = price
            test_pd["income"] = opt_gain
            test_pd["before_income"] = before_gain

            # 去掉未竞得的和数据异常的
            test_pd = test_pd[test_pd.target_price > 0]
            test_pd = test_pd[test_pd.target_price <= test_pd.response_ecpm]

            # 上帝视角搜索最优价格
            total_gain = 0
            best_raio = 0
            for ratio in np.arange(0, 1, 0.01):
                test_pd["bfhp_price"] = ratio * test_pd["response_ecpm"]
                test_pd["label_br"] = np.select([
                    (test_pd["bfhp_price"] >= test_pd["target_price"])],
                    [test_pd["response_ecpm"] - test_pd["bfhp_price"]], default=0)
                now_gain = np.sum(test_pd["label_br"])
                if now_gain > total_gain:
                    total_gain = now_gain
                    best_raio = ratio

            test_pd["bfhp_price"] = best_raio * test_pd["response_ecpm"]
            test_pd["label_br"] = np.select([
                (test_pd["bfhp_price"] >= test_pd["target_price"])],
                [test_pd["bfhp_price"]], default=0)

            test_pd["label_before"] = np.select([
                (test_pd["response_ecpm"] >= test_pd["target_price"])],
                [test_pd["response_ecpm"]], default=0)
            """
            test_pd["label_increase"] = np.select([
                (test_pd["virtual_ecpm"] >= test_pd["target_price"])],
                [test_pd["virtual_ecpm"]], default=0)
            """
            test_pd["label_after"] = np.select([
                (test_pd["shading_ecpm"] >= test_pd["target_price"])],
                [test_pd["shading_ecpm"]], default=0)

            self.shading_result_list.append(test_pd)

            # 竞得率 & cpm
            win_rate_before = win_rate_after = 1.0
            cpm_after = 0
            if len(test_pd["label_before"]) != 0:
                win_rate_before = len(test_pd[test_pd["label_before"] > 0]) / len(test_pd["label_before"])
            cpm_before = sum(test_pd["label_before"]) / len(test_pd[test_pd["label_before"] > 0])
            gain_before = np.sum(test_pd[test_pd.label_before > 0]["response_ecpm"]
                                 - test_pd[test_pd.label_before > 0]["label_before"])

            if len(test_pd[test_pd["label_br"] > 0]):
                win_rate_br = len(test_pd[test_pd["label_br"] > 0]) / len(test_pd["label_br"])
                cpm_br = sum(test_pd["label_br"]) / len(test_pd[test_pd["label_br"] > 0])

            if len(test_pd[test_pd["label_after"] > 0]):
                win_rate_after = len(test_pd[test_pd["label_after"] > 0]) / len(test_pd["label_after"])
                cpm_after = sum(test_pd["label_after"]) / len(test_pd[test_pd["label_after"] > 0])
                gain_after = np.sum(test_pd[test_pd.label_after > 0]["response_ecpm"]
                                    - test_pd[test_pd.label_after > 0]["label_after"])

            self.logging.info(f"key {key} \n"
                              f"win rate br: {win_rate_br}, cpm br: {cpm_br}, gain_br: {total_gain} \n"
                              f"win rate before: {win_rate_before}, cpm before: {cpm_before}, gain before: {gain_before} \n"
                              f"win rate after: {win_rate_after}, cpm after: {cpm_after}, gain after: {gain_after}")

            evaluation[key] = {
                "win_rate_before": win_rate_before,
                "cpm_before": cpm_before,
                "win_rate_after": win_rate_after,
                "cpm_after": cpm_after
            }

        return evaluation

    def result_metrics(self, data_pd, label_name):
        """
        输入含label的dataframe，输出需要的指标的计算结果
        输出指标按顺序为:
            win rate
            cpm
            rr
            revenue
            surplus
            price_elasticity
        """
        tmp_pd = data_pd[data_pd[label_name] > 0]
        if tmp_pd.empty:
            return 0, 0, 0, 0, 0, 0

        cpm_all = np.mean(data_pd["response_ecpm"])
        win_rate = len(tmp_pd) / len(data_pd)
        cpm = np.mean(tmp_pd[label_name])
        rr = np.sum(tmp_pd["response_ecpm"] - tmp_pd[label_name]) \
             / np.sum(tmp_pd["response_ecpm"] - tmp_pd["win_price"])
        revenue = np.sum(tmp_pd["win_price"])

        surplus = np.sum(tmp_pd["response_ecpm"] - tmp_pd[label_name])

        price_elasticity = 0
        if cpm_all - cpm != 0:
            price_elasticity = (1 - win_rate) * cpm_all / (cpm_all - cpm)

        return win_rate, cpm, rr, revenue, surplus, price_elasticity

    def result_evaluation_steps(self, result):
        """
        评估search
        输入 {key: {loop_index: { } }
        """
        evaluation = {}
        self.logging.info(f"result.keys():{result.keys()}\n"
                          f"test_dataset:{self.data_list.head(100)}")

        for key, dict in result.items():
            evaluation[key] = {}
            test_pd = self.data_list[self.data_list.key == key]
            # 去掉未竞得的和数据异常的
            test_pd = test_pd[test_pd.target_price > 0]
            test_pd = test_pd[test_pd.target_price <= test_pd.response_ecpm]

            if test_pd.empty:
                self.logging.info(f"key:{key}, test_pd is empty!")
                continue
            else:
                self.logging.info(f"key:{key}, test_pd data nums: {len(test_pd)}")

            # 上帝视角搜索最优价格，以最大化surplus为目标
            max_surplus = 0
            best_raio = 0
            for ratio in np.arange(0, 1, 0.01):
                test_pd["br_price"] = ratio * test_pd["response_ecpm"]
                test_pd["surplus_br"] = np.select([
                    (test_pd["br_price"] >= test_pd["target_price"])],
                    [test_pd["response_ecpm"] - test_pd["br_price"]], default=0)
                now_surplus = np.sum(test_pd["surplus_br"])
                if now_surplus > max_surplus:
                    max_surplus = now_surplus
                    best_raio = ratio

            test_pd["br_price"] = best_raio * test_pd["response_ecpm"]
            test_pd["label_br"] = np.select([
                (test_pd["br_price"] >= test_pd["target_price"])],
                [test_pd["br_price"]], default=0)

            win_rate_br, cpm_br, rr_br, revenue_br, surplus_br, price_elasticity_br = self.result_metrics(test_pd,
                                                                                                          "label_br")
            surplus_upper_bound = np.sum(test_pd["response_ecpm"] - test_pd["target_price"])
            cpm_win_price = np.mean(test_pd["target_price"])
            revenue_upper_bound = np.sum(test_pd["target_price"])

            record_nums = len(dict)
            # point_step = record_nums // EVALUATION_POINT_NUMS
            # if point_step == 0:
            #     continue

            for index, result_dict in dict.items():
                if 'true' in str(index) or float(index) % (MAB_SAVE_STEP * EVALUATION_POINT_STEP) != 0:
                    continue

                market_price = result_dict["market_price"]
                chosen_count_map = result_dict["chosen_count_map"]
                imp_count_map = result_dict["imp_count_map"]
                norm_dict = result_dict["norm_dict"]
                norm_max = float(norm_dict["norm_max"])
                norm_min = float(norm_dict["norm_min"])
                price, opt_gain, before_gain \
                    = self.cal_price_adjustment_gain.get_adjust_price(
                        test_pd[["response_ecpm", "win_price", "click_num",
                                 "target_cpa", "pay_amount"]],
                        market_price, chosen_count_map,
                        imp_count_map, norm_dict)

                # 二价数据集测试效果
                test_pd = test_pd.copy()
                test_pd["shading_ecpm"] = price
                test_pd["income"] = opt_gain
                test_pd["before_income"] = before_gain

                test_pd["label_before"] = np.select([
                    (test_pd["response_ecpm"] >= test_pd["target_price"])],
                    [test_pd["response_ecpm"]], default=0)

                test_pd["label_mab"] = np.select([
                    (test_pd["shading_ecpm"] >= test_pd["target_price"])],
                    [test_pd["shading_ecpm"]], default=0)

                win_rate_before, cpm_before, rr_before, revenue_before, surplus_before, price_elasticity_before = self.result_metrics(
                    test_pd,
                    "label_before")
                win_rate_mab, cpm_mab, rr_mab, revenue_mab, surplus_mab, price_elasticity_mab = self.result_metrics(
                    test_pd,
                    "label_mab")

                evaluation[key][index] = {
                    "rr_mab": rr_mab,
                    "rr_br": rr_br,
                    "win_rate_before": win_rate_before,
                    "win_rate_br": win_rate_br,
                    "win_rate_mab": win_rate_mab,
                    "cpm_before": cpm_before,
                    "cpm_br": cpm_br,
                    "cpm_mab": cpm_mab,
                    "cpm_win_price": cpm_win_price,
                    "price_elasticity_br": price_elasticity_br,
                    "price_elasticity_mab": price_elasticity_mab,
                    "revenue_upper_bound": revenue_upper_bound,
                    "revenue_mab": revenue_mab,
                    "revenue_br": revenue_br,
                    "surplus_upper_bound": surplus_upper_bound,
                    "surplus_mab": surplus_mab,
                    "surplus_br": surplus_br
                }

        return evaluation

    def do_process(self, bandit_dict, method_name):

        # 步骤一、获取测试数据
        self.read_data()

        # 步骤二、评估结果
        evaluation_dict = self.result_evaluation_steps(bandit_dict)

        result_dir = f"./result/{method_name}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        mhour = datetime.now().strftime("%Y%m%d%H")
        with open(result_dir + f"/evaluation_result_{mhour}_{self.TEST_DATA_PATH[7:-4]}_{method_name}.json", mode='w', encoding='utf-8') as f:
            json.dump(evaluation_dict, f)


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d ]in %(funcName)-8s  %(message)s"
    )

    rd = ReadData(logging)
    rd.data_process()
