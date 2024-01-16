import collections


def search_price_for_optimal_cost(logging, ecpm, market_price, chosen_count_map, imp_count_map, norm_dict):
    opt_price = 1.0
    opt_gain = 0
    before_gain = 0
    norm_max = float(norm_dict["norm_max"])
    norm_min = float(norm_dict["norm_min"])
    win_rate_dict = {}
    for price, chosen_count in chosen_count_map.items():
        if chosen_count < 1 or price not in imp_count_map:
            continue

        imp_count = imp_count_map[price]

        price = float(price) * (norm_max - norm_min) + norm_min

        win_rate = imp_count * 1.0 / chosen_count
        win_rate_dict[price] = win_rate
        expect_gain = win_rate * (ecpm - price)
        if expect_gain > opt_gain:
            opt_gain = expect_gain
            opt_price = price

    opt_price = round(opt_price, 4)

    return opt_price, opt_gain, before_gain


def search_price_for_optimal_cost_win_rate(logging, ecpm, market_price, win_rate_map):
    opt_price = 1.0
    opt_gain = 0
    before_gain = 0
    win_rate_dict = {}
    for price, win_rate in win_rate_map.items():
        price = float(price)
        win_rate = float(win_rate)

        win_rate_dict[price] = win_rate
        expect_gain = win_rate * (ecpm - price)
        if expect_gain > opt_gain:
            opt_gain = expect_gain
            opt_price = price

    opt_price = round(opt_price, 4)

    return opt_price, opt_gain, before_gain


def search_price_for_optimal_income(logging, ecpm, market_price, gmv, chosen_count_map,
                                    imp_count_map, norm_dict):
    opt_price = 1.0
    opt_gain = 0
    before_gain = 0
    norm_max = norm_dict["norm_max"]
    norm_min = norm_dict["norm_min"]
    gap = ecpm
    win_rate_dict = {}
    for price, chosen_count in chosen_count_map.items():
        if chosen_count < 1 or price not in imp_count_map:
            continue

        imp_count = imp_count_map[price]
        price = price * (norm_max - norm_min) + norm_min
        win_rate = imp_count * 1.0 / chosen_count
        win_rate_dict[price] = win_rate
        expect_gain = win_rate * (gmv - price)
        if gap > abs(price - ecpm):
            gap = abs(price - ecpm)
            before_gain = win_rate * (gmv - ecpm)

        if expect_gain > opt_gain:
            opt_gain = expect_gain
            opt_price = price

    opt_price = round(opt_price, 4)

    return opt_price, opt_gain, before_gain
