import json
import numpy as np
import pandas as pd
from configs.config import DATA_NUMS_LOWER_BOUND, BIN_NUMS


class ReadData(object):

    def __init__(self, logging, data_path):

        self.logging = logging
        self.data_path = data_path

    def read_csv_data(self):

        df = pd.read_csv(self.data_path, sep="\t")
        data_pd = df.astype({
            'tdbank_imp_date': np.str
            , 'media_app_id': np.int64
            , 'position_id': np.int64
            , 'pltv': np.int64
            , 'pctcvr': np.float64
            , 'pctr': np.float
            , 'bid_price': np.float
            , 'response_ecpm': np.float
            , 'win_price': np.float
            , 'winner_bid_price': np.float
            , 'click_num': np.int64
            , 'target_cpa': np.float
            , 'pay_amount': np.float
        })

        self.logging.info(f"data_pd:{data_pd.head(10)}")
        return data_pd

    def data_filter(self, data_pd):
        tmp = data_pd[data_pd['response_ecpm'] > 0]
        win_price_list = np.array(tmp['response_ecpm'])
        per_95 = np.percentile(win_price_list, 95)
        self.logging.info(f"pre_95:{per_95}, median:{np.median(win_price_list)}")
        data_pd = data_pd[data_pd['response_ecpm'] <= per_95]

        self.logging.info(f"data_pd:{data_pd.head(10)}")

        return data_pd

    def data_discret_norm(self, data_pd):
        data_pd = data_pd.copy()
        data_pd["key"] = data_pd["media_app_id"].map(str) \
            .str.cat([data_pd["position_id"].map(str)], sep="_")
        norm_pd_list = []
        for key, group_pd in data_pd.groupby(["key"]):
            if len(group_pd) < DATA_NUMS_LOWER_BOUND:
                continue

            group_pd["norm_min"] = 0
            group_pd["norm_max"] = group_pd["response_ecpm"].max()
            group_pd["norm_ecpm"] = (group_pd["response_ecpm"] - group_pd["norm_min"]) \
                                    / (group_pd["norm_max"] - group_pd["norm_min"])

            bins = pd.qcut(group_pd["norm_ecpm"], q=BIN_NUMS, retbins=True)[1]
            bins[0] = 0

            group_pd["interval"] = pd.qcut(group_pd["norm_ecpm"], q=BIN_NUMS)
            group_pd["interval_index"] = pd.qcut(group_pd["norm_ecpm"], q=BIN_NUMS, labels=False)
            group_pd["bins"] = json.dumps(list(bins))

            norm_pd_list.append(group_pd)

        data_pd = pd.concat(norm_pd_list)

        data_pd["win_price"] = (data_pd["win_price"] - data_pd["norm_min"]) / (data_pd["norm_max"] - data_pd["norm_min"])
        data_pd["winner_bid_price"] = (data_pd["winner_bid_price"] - data_pd["norm_min"]) / (data_pd["norm_max"] - data_pd["norm_min"])

        data_pd["response_ecpm"] = data_pd["interval"].map(lambda x: x.right)
        data_pd["interval_index"] = data_pd["interval_index"].map(int)
        return data_pd

    def get_data_dict_struct(self, data_pd, is_test=False):
        response_dict = {}
        for index, row in data_pd.iterrows():
            media_app_id = int(row["media_app_id"])
            position_id = int(row["position_id"])
            value = float(row["response_ecpm"])

            if media_app_id not in response_dict:
                response_dict[media_app_id] = {}

            position_dict = response_dict[media_app_id]
            if position_id not in position_dict:
                position_dict[position_id] = []
            if is_test:
                position_dict[position_id].append(row)
            else:
                position_dict[position_id].append(value)

            response_dict[media_app_id] = position_dict

        return response_dict

    def data_process(self):
        """
        main function
        """
        market_price_dict = {}
        norm_dict = {}

        data_pd = self.read_csv_data()
        data_pd = self.data_filter(data_pd)
        data_pd = self.data_discret_norm(data_pd)

        imp_dict = self.get_data_dict_struct(data_pd[data_pd['win_price'] > 0])
        no_imp_dict = self.get_data_dict_struct(data_pd[data_pd['win_price'] == 0])

        for media_app_id, position_info in imp_dict.items():
            if media_app_id not in market_price_dict:
                market_price_dict[media_app_id] = {}

            position_dict = market_price_dict[media_app_id]
            for position_id, value_list in position_info.items():
                position_dict[position_id] = np.median(np.array(value_list))

            market_price_dict[media_app_id] = position_dict

        for key, group_pd in data_pd.groupby(["key"]):
            [media_app_id, position_id] = map(int, key.split("_"))
            write_line = group_pd.iloc[0]
            if media_app_id not in norm_dict:
                norm_dict[media_app_id] = {}

            group_pd["market_price"] = np.select(
                [group_pd["win_price"] > group_pd["winner_bid_price"]],
                [group_pd["win_price"]], default=group_pd["winner_bid_price"]
            )

            cut_bins = pd.cut(group_pd["market_price"], bins=json.loads(write_line["bins"]))

            norm_dict[media_app_id][position_id] = {
                "norm_min": float(write_line["norm_min"]),
                "norm_max": float(write_line["norm_max"]),
                "bins": write_line["bins"],
                "market_price_list": json.dumps(list(group_pd["market_price"].groupby(cut_bins).count()))
            }

        self.logging.info(f"len imp_dict:{len(imp_dict)},  len no_imp_dict:{len(no_imp_dict)}, "
                          f"len market_price_dict:{len(market_price_dict)}")
        return market_price_dict, imp_dict, no_imp_dict, norm_dict, data_pd

    def test_data_process(self):
        """
        read test dataset
        """
        data_pd_test = self.read_csv_data()

        data_pd_test = self.data_filter(data_pd_test)

        data_pd_test["key"] = data_pd_test["media_app_id"].map(str) \
            .str.cat([data_pd_test["position_id"].map(str)], sep="_")
        data_pd_test["target_price"] = data_pd_test[["win_price", "winner_bid_price"]].T.max()

        return data_pd_test[["key", "response_ecpm", "target_price", "win_price", "click_num",
                             "target_cpa", "pay_amount"]]

    def read_test_data_process(self, norm_dict):
        """
        read test dataset
        """
        data_pd = self.read_csv_data()
        data_pd = self.data_filter(data_pd)
        norm_pd_list = []
        for media_app_id in norm_dict:
            for position_id in norm_dict[media_app_id]:
                norm_min = norm_dict[media_app_id][position_id]["norm_min"]
                norm_max = norm_dict[media_app_id][position_id]["norm_max"]
                bins = json.loads(norm_dict[media_app_id][position_id]["bins"])

                group_pd = data_pd[(data_pd.media_app_id == media_app_id) & (data_pd.position_id == position_id)].copy()

                group_pd["norm_ecpm"] = (group_pd["response_ecpm"] -norm_min) / (norm_max - norm_min)
                group_pd["win_price"] = (group_pd["win_price"] - norm_min) / (norm_max - norm_min)

                group_pd["interval"] = pd.cut(group_pd["norm_ecpm"], bins=bins)
                group_pd = group_pd.dropna()

                norm_pd_list.append(group_pd)

        data_pd = pd.concat(norm_pd_list)
        data_pd["response_ecpm"] = data_pd["interval"].map(lambda x: x.right)

        return data_pd


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d ]in %(funcName)-8s  %(message)s"
    )

    rd = ReadData(logging)
    rd.data_process()
