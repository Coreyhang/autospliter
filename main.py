from utils import param_cal, save_csv, read_csv
from gpt_spliter import GPTSpliter
from draw import draw
from config import *
import os


def main():
    for name, config in gpt_config_all():
        # parameters of GPT
        print('############## This is ', name)
        n_layers, d_model, n_heads, d_head = config

        param_count = param_cal(n_layers=n_layers, n_heads=n_heads, d_head=d_head, d_model=d_model)
        print("## Parameter Number is : {:.3f} M".format(param_count / 1000000))

        spliter = GPTSpliter(n_layers=n_layers, n_heads=n_heads, d_head=d_head, d_model=d_model, T=1000,
                             solution_num=300)
        # draw(spliter.lut)
        # draw(spliter.lut, save=(name + '_t1000'))
        # save_csv(spliter.lut, name + '_t1000')
        # os.system('zip /data/sch/gpt_lut/' + name + '_t1000.zip /data/sch/gpt_lut/' + name + '_t1000.csv')
        # os.system('rm /data/sch/gpt_lut/' + name + '_t1000.csv')


if __name__ == '__main__':
    main()
