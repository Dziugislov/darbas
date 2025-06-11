import read_ts

all_data = read_ts.read_ts_ohlcv_dat('A_OHLCV_@AD_minutes_1440_1703-1557.dat')

#tick_size
tick_size = all_data[0].big_point_value * all_data[0].tick_size
ohlc_data = all_data[0].data