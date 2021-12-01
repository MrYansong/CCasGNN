
from CCasGNN import CCasGNN_Trainer
from utils import Logger
from param_parser import parameter_parser
import sys
import time

def main():
    start = time.time()
    args = parameter_parser()
    sys.stdout = Logger(args.result_log)
    model = CCasGNN_Trainer(args)
    model.fit()
    # model.test()
    end = time.time()
    print('consume ', (end - start) / 60, ' minutes')

if __name__ == '__main__':
    main()