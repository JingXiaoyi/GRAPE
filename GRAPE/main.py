import argparse
import time
from recbole.quick_start import run_recbole


if __name__ == '__main__':

    begin = time.time()
    parameter_dict = {
        'neg_sampling': None,
        # 'gpu_id':3,
        # 'attribute_predictor':'not',
        # 'attribute_hidden_size':"[256]",
        # 'fusion_type':'gate',
        # 'seed':212,
        # 'n_layers':4,
        # 'n_heads':1
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRAPE', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Green_Food_Rec', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='configs/Green_Food_Rec.yaml', help='config files')

    parser.add_argument('--ada_fuse', type=int, default=1)   # adaptive learnable att fusion weight

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--item_predictor', type=int, default=2)
    parser.add_argument('--ip_gate_mode', type=str, default='moe')
    parser.add_argument('--gate_drop', type=int, default=1)

    parser.add_argument('--clayer_num', type=int, default=1)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--n_heads', type=int)

    parser.add_argument('--aap', type=str, default='wi_wc_bce')
    parser.add_argument('--app_gate', type=int, default=1)
    parser.add_argument('--lamdas', type=int, default=5)

    parser.add_argument('--logit_num', type=int, default=2)
    parser.add_argument('--pooling_mode', type=str, default='sum')

    parser.add_argument('--seqmc', type=str, default='all')
    parser.add_argument('--cdmc', type=str, default='all')
    parser.add_argument('--sc', type=int, default=1)

    parser.add_argument('--ssl', type=int, default=1)
    parser.add_argument('--cl', type=str, default='idropwc')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--cllmd', type=float, default=0.1)
    parser.add_argument('--sim', type=str, default='dot')

    parser.add_argument('--pred', type=str, default='dot')

    parser.add_argument('--mask_ratio', type=float, default=0.2)

    parser.add_argument('--result_file', type=str)
    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_result = run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
    end = time.time()
    print(end-begin)

    with open(args.result_file, 'a+') as f:
        f.write('model:' + str(run_result['model']) + '\n')
        f.write('valid result:' + str(run_result['best_valid_result']) + '\n')
        f.write('test result:' + str(run_result['test_result']) + '\n')
        f.write('\n')
