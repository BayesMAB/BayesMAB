import os

from multiprocessing import Pool
import multiprocessing

import datetime
from data_process.read_data import ReadData
from data_process.result_evaluate import ResultEvaluate
from configs.config import parallel_num, Environment, Multi_Process, search_test
from bandit import *
import logging
import json
from registry import registry
import numpy as np
import pandas as pd
import argparse


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


class BidShading(object):
    """
    bid shading 主类
    """

    def __init__(self, logging, mdate, method_name):
        """
        初始化
        """
        self.logging = logging

        # 价格数据初始化
        self.market_price_dict = {}
        self.impression_price_dict = {}
        self.media_position_dict = {}
        self.no_impression_price_dict = {}
        self.norm_dict = {}
        self.test_imp_dict = {}

        self.optimal_ratio_dict = {}
        self.mdate = mdate

        self.method_class = registry.get(method_name)
        if self.method_class is None:
            raise Exception("Invalid version!")

        self.method_name = method_name

    def get_data_path(self):
        mdate = self.mdate

        date = datetime.datetime(int(mdate[0:4]), int(mdate[4:6]), int(mdate[6:8]))
        valid_date = mdate
        train_date = date + datetime.timedelta(days=1)
        test_date = date + datetime.timedelta(days=2)

        # self.VALID_DATA_PATH = "./data/bid_shading_{0}09_{0}20.txt".format(valid_date)
        # self.TRAIN_DATA_PATH = "./data/bid_shading_{0}09_{0}20.txt".format(train_date.strftime('%Y%m%d'))
        # self.TEST_DATA_PATH = "./data/bid_shading_{0}09_{0}20.txt".format(test_date.strftime('%Y%m%d'))
        self.VALID_DATA_PATH = "./data/data_{0}.txt".format(valid_date)
        self.TRAIN_DATA_PATH = "./data/data_{0}.txt".format(train_date.strftime('%Y%m%d'))
        self.TEST_DATA_PATH = "./data/data_{0}.txt".format(test_date.strftime('%Y%m%d'))

        self.logging.info("load data from valid {} train {} test {}".format(self.VALID_DATA_PATH, self.TRAIN_DATA_PATH, self.TEST_DATA_PATH))

    def read_data(self):
        # 1、读取 bid shading输入数据
        rd = ReadData(logging=self.logging, data_path=self.VALID_DATA_PATH)

        # market_price_dict = media_app_id:position_id:pltv - value
        # impression_price_dict = media_app_id:position_id:pltv - value_list
        self.market_price_dict, self.impression_price_dict, self.no_impression_price_dict,\
            self.norm_dict, self.data_pd = rd.data_process()

        self.logging.info(f"len market_price_dict:{len(self.market_price_dict)}, "
                          f"len impression_price_dict:{len(self.impression_price_dict)}")

        # 2、获取media、position映射关系
        for media_id, position_info in self.market_price_dict.items():
            if media_id not in self.media_position_dict:
                self.media_position_dict[media_id] = set(position_info.keys())
            else:
                self.media_position_dict[media_id] = self.media_position_dict[media_id] | set(position_info.keys())

        for media_id, position_info in self.impression_price_dict.items():
            if media_id not in self.media_position_dict:
                self.media_position_dict[media_id] = set(position_info.keys())
            else:
                self.media_position_dict[media_id] = self.media_position_dict[media_id] | set(position_info.keys())

        self.logging.info(f"media_position_dict:{self.media_position_dict}")

    def remove_local_backend(self):
        """
        删除临时文件
        :return:
        """
        return

    def run(self):
        self.get_data_path()
        self.logging.info("run -> start")

        bandit = self.method_class()
        re = ResultEvaluate(self.logging, self.TEST_DATA_PATH)

        if not search_test:
            # 1、读取相关要处理的的数据（valid data）
            self.read_data()

            rd = ReadData(logging=self.logging, data_path=self.TRAIN_DATA_PATH)
            self.data_pd = rd.read_test_data_process(self.norm_dict)

            # 2、计算并保存数据
            if not Multi_Process:
                for media_app_id in self.media_position_dict.keys():
                    res = bandit.do_process(media_app_id, self.media_position_dict, self.market_price_dict,
                                            self.impression_price_dict, self.no_impression_price_dict,
                                            self.norm_dict[media_app_id], self.data_pd)
                    self.optimal_ratio_dict.update(res)  # 保存进程返回结果
            else:
                pool = Pool(parallel_num)  # 构建进程池
                res_l = []  # 保存进程返回结果
                mgr = multiprocessing.Manager()
                media_position_dict_obj = mgr.dict(self.media_position_dict)
                market_price_dict_obj = mgr.dict(self.market_price_dict)
                impression_price_dict_obj = mgr.dict(self.impression_price_dict)
                no_impression_obj = mgr.dict(self.no_impression_price_dict)

                for media_app_id in self.media_position_dict.keys():
                    norm_dict = mgr.dict(self.norm_dict[media_app_id])
                    res = pool.apply_async(bandit.do_process,
                                           args=(media_app_id, media_position_dict_obj, market_price_dict_obj,
                                                 impression_price_dict_obj, no_impression_obj, norm_dict))
                    res_l.append(res)

                pool.close()  # 关闭进程池，不再接受请求
                pool.join()  # 等待所有的子进程结束

                for res in res_l:
                    self.logging.info(f"res:{res}")
                    self.optimal_ratio_dict.update(res.get())

            # 保存结果
            result_dir = f"./result/{self.method_name}"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            mhour = datetime.datetime.now().strftime("%Y%m%d%H")
            with open(result_dir + f"/bandit_result_{mhour}_{self.TEST_DATA_PATH[7:-4]}_{self.method_name}.json", mode='w',
                      encoding='utf-8') as f:
                json.dump(self.optimal_ratio_dict, f)

        else:
            try:
                all_files, all_dirs = [], []

                for root, dirs, files in os.walk('./result'):
                    for file in files:
                        all_files.append(os.path.join(root, file))

                    for dir in dirs:
                        all_dirs.append(os.path.join(root, dir))

                train_result_path = ""
                for file in all_files:
                    if "evaluation_result" in file and self.method_name+'.json' in file and "2022102309" in file:
                        os.remove(file)
                        print(f"succeed remove file {file}")
                    if "bandit_result" in file and self.method_name+'.json' in file and "2022102309" in file:
                        train_result_path = file

            except:
                raise Exception("error!")

        self.logging.info(f"run -> end len(optimal_ratio_dict):{len(self.optimal_ratio_dict)}")
        # self.logging.info(f"output optimal_ratio_dict:{self.optimal_ratio_dict}")

        re.do_process(self.optimal_ratio_dict, self.method_name)

        # 3、计算完 删除临时文件
        self.remove_local_backend()


def main(mdate, method_name):
    bs = BidShading(logging, mdate, method_name)
    bs.run()


if __name__ == '__main__':
    # python3 bid_shading_e_e.py > train.log 2>&1 &
    parser = argparse.ArgumentParser(description="bid shading experiment")
    # add args
    parser.add_argument("-mdate", default="20221019", type=str,
                        help="date of data, record of the chosen date will be Valid Data, "
                             "record of the next date will be Train Data")
    parser.add_argument("-method_name", default="UCBNetwork", type=str,
                        choices=["BayesMAB", "UCB1", "UCB2", "UCBBanditNoPrior", "UCBBanditIndependent",
                                 "MOSS", "EpsilonGreedyBandit", "ThompsonSamplingBandit", "UCBNetwork"],
                        help="MAB algorithm to use")
    args = parser.parse_args()
    print(args)

    main(args.mdate, args.method_name)
    