import time
import tushare as ts

for code in ['000513', '300078', 'sz', 'cyb', 'sh', '601318', '600276', '600519', '000651']:
    print(code)
    time.sleep(0.3)
    data = ts.get_k_data(code, ktype='w', autype='qfq', index=False,
                         start='2001-01-01', end=time.strftime("%Y-%m-%d"))
    data.to_csv(code + "_w_ori.csv")
    data.to_csv(code + "_w_bas.csv")
    time.sleep(0.3)
    data = ts.get_k_data(code, ktype='D', autype='qfq', index=False,
                        start='2001-01-01', end=time.strftime("%Y-%m-%d"))
    data.to_csv(code + "_d_ori.csv")
    data.to_csv(code + "_d_bas.csv")