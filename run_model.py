import argparse
import time
from recbole.quick_start import run_recbole


if __name__ == '__main__':

    begin = time.time()
    parameter_dict = {
        'neg_sampling': None,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='GRAPE', help='name of models')

    parser.add_argument('--dataset', '-d', type=str, default='Green_Rec', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='configs/Green_Rec.yaml', help='config files')
    parser.add_argument('--priority', type=int, default=0, help='Loss Type')
    # Jing
    # 1: ENH
    # 2: EHN
    # 3: HEN
    # 4: HNE
    # 5: NEH
    # 6: NHE
    parser.add_argument('--green_alpha', type=float, default=0.8, help='Alpha')
    parser.add_argument('--green_beta_e', type=float, default=80, help='Beta-EIS')
    parser.add_argument('--green_beta_n', type=float, default=35, help='Beta-NIS')
    parser.add_argument('--green_beta_m', type=float, default=35, help='Beta-HMI')

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--pooling_mode', type=str, default='mean')
    
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=2)

    parser.add_argument('--ada_fuse', type=int, default=1)   # adaptive learnable att fusion weight
    parser.add_argument('--ip_mode', type=str, default='gating')

    parser.add_argument('--aaplmd', type=int, default=4)
    parser.add_argument('--aap', type=str, default='wi_wc_bce')
    parser.add_argument('--app_gate', type=int, default=1)
    parser.add_argument('--attribute_predictor', type=str, default='linear')

    parser.add_argument('--ssl', type=int, default=1)
    parser.add_argument('--cl', type=str, default='idropwc')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--cllmd', type=float, default=0.14)
    parser.add_argument('--sim', type=str, default='dot')


    parser.add_argument('--fusion_type', type=str, default='gate')
    parser.add_argument('--attribute_hidden_size', type=int, default=2048)
    parser.add_argument('--train_batch_size', type=list, default=[256])

    parser.add_argument('--result_file', type=str, default='./green.out')
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
