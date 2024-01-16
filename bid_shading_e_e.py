import os
import json
import logging
import bandit
import datetime
import argparse
import multiprocessing
from registry import registry
from multiprocessing import Pool

from data_process.read_data import ReadData
from data_process.result_evaluate import ResultEvaluate
from configs.config import parallel_num, Environment, Multi_Process


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

    def __init__(self, logging, mdate, method_name):
        self.logging = logging

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

        self.VALID_DATA_PATH = "./data/data_{0}.txt".format(valid_date)
        self.TRAIN_DATA_PATH = "./data/data_{0}.txt".format(train_date.strftime('%Y%m%d'))
        self.TEST_DATA_PATH = "./data/data_{0}.txt".format(test_date.strftime('%Y%m%d'))

        self.logging.info("load data from valid {} train {} test {}".format(self.VALID_DATA_PATH, self.TRAIN_DATA_PATH, self.TEST_DATA_PATH))

    def read_data(self):
        rd = ReadData(logging=self.logging, data_path=self.VALID_DATA_PATH)

        self.market_price_dict, self.impression_price_dict, self.no_impression_price_dict,\
            self.norm_dict, self.data_pd = rd.data_process()

        self.logging.info(f"len market_price_dict:{len(self.market_price_dict)}, "
                          f"len impression_price_dict:{len(self.impression_price_dict)}")

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

    def run(self):
        self.get_data_path()
        self.logging.info("run -> start")

        bandit = self.method_class()
        re = ResultEvaluate(self.logging, self.TEST_DATA_PATH)

        self.read_data()

        rd = ReadData(logging=self.logging, data_path=self.TRAIN_DATA_PATH)
        self.data_pd = rd.read_test_data_process(self.norm_dict)

        if not Multi_Process:
            for media_app_id in self.media_position_dict.keys():
                res = bandit.do_process(media_app_id, self.media_position_dict, self.market_price_dict,
                                        self.impression_price_dict, self.no_impression_price_dict,
                                        self.norm_dict[media_app_id], self.data_pd)
                self.optimal_ratio_dict.update(res)
        else:
            pool = Pool(parallel_num)
            res_l = []
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

            pool.close()
            pool.join()

            for res in res_l:
                self.logging.info(f"res:{res}")
                self.optimal_ratio_dict.update(res.get())

        result_dir = f"./result/{self.method_name}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        mhour = datetime.datetime.now().strftime("%Y%m%d%H")
        with open(result_dir + f"/bandit_result_{mhour}_{self.TEST_DATA_PATH[7:-4]}_{self.method_name}.json", mode='w',
                  encoding='utf-8') as f:
            json.dump(self.optimal_ratio_dict, f)

        self.logging.info(f"run -> end len(optimal_ratio_dict):{len(self.optimal_ratio_dict)}")

        re.do_process(self.optimal_ratio_dict, self.method_name)


def main(mdate, method_name):
    bs = BidShading(logging, mdate, method_name)
    bs.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="bid shading experiment")
    # add args
    parser.add_argument("-mdate", default="20221019", type=str,
                        help="date of data, record of the chosen date will be Valid Data, "
                             "record of the next date will be Train Data")
    parser.add_argument("-method_name", default="BayesMAB", type=str,
                        choices=["BayesMAB", "UCB1", "UCB2", "UCBBanditNoPrior", "UCBBanditIndependent",
                                 "MOSS", "EpsilonGreedyBandit", "ThompsonSamplingBandit", "UCBNetwork",
                                 "BayesUCB", "HierarchicalThompsonSamplingBandit"],
                        help="MAB algorithm to use")
    args = parser.parse_args()
    print(args)

    main(args.mdate, args.method_name)
    