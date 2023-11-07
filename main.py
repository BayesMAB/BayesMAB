import logging
from bid_shading_e_e import BidShading
# from tools.reward_ratio_result_plot import mean_plot_main
import argparse


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
    # parser.add_argument("-method_name", default="BayesMAB", type=str,
    #                     choices=["BayesMAB", "UCB1", "UCB2", "UCBBanditNoPrior", "UCBBanditIndependent",
    #                              "MOSS", "EpsilonGreedyBandit", "ThompsonSamplingBandit", "UCBNetwork",
    #                              "HierarchicalThompsonSamplingBandit"],
    #                     help="MAB algorithm to use")
    parser.add_argument("-method_name", default="HierarchicalThompsonSamplingBandit", type=str,
                        choices=["BayesMAB", "UCB1", "UCB2", "UCBBanditNoPrior", "UCBBanditIndependent",
                                 "MOSS", "EpsilonGreedyBandit", "ThompsonSamplingBandit", "UCBNetwork",
                                 "HierarchicalThompsonSamplingBandit", "BayesUCB"],
                        help="MAB algorithm to use")
    args = parser.parse_args()
    print(args)

    for mdate in ["20221017", "20221018", "20221019", "20221020", "20221021"]:
        main(mdate, args.method_name)

    # mean_plot_main(["BayesMAB", "UCBNetwork"])