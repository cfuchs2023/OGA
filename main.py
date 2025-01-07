# -*- coding: utf-8 -*-
import argparse
import runner

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_path', type=str)
    parser.add_argument('-d', '--datasets', default = ['dtd'], nargs="*", help = 'List of datasets to adapt on.')
    parser.add_argument('--adapt_method_name', default='OGA', type=str,
                        help='The name of the online adaption method to use. One of TDA, DMN or OGA.')
    parser.add_argument('--backbone', default='vit_b16', type=str, help = 'Name of the backbone to use. Examples : vit_b16 or rn101.')
    parser.add_argument('--root_cache_path', default = None, type = str, help = 'Root path of cached features and targets. Features and target should be located at root_cache_path/{dataset}/cache. Defaults to data_root_path internally.')
    parser.add_argument('--root_prompts_path', default = None, type = str, help = 'Path where learned prompts should be located when using coop or taskres.') #TODO: clarify
    parser.add_argument('--root_save_path', default = None, type = str, help = 'Path where results are stored. Defaults to data_root_path/results internally.')
    parser.add_argument('--run_name', default = None, type = str, help = 'Name of the results files. Has an internal default value and can be left to None. The code outputs two results files, {run_name}.json and {run_name}.pickle.')
    parser.add_argument('--n_runs', default = 100, type = int, help = 'Number of runs for each dataset.')
    parser.add_argument('--batch_size', default = 32, type = int)
    parser.add_argument('--shot_capacity', default = 8, type = int, help = 'Maximum size of the memory of the online adaptation method, expressed in number of shots per class.')
    parser.add_argument('--check_on_fullset' , default = False, type = int, help = 'Wether to evaluate the accuracy of the adapted model on the complete test set at regular intervals. Results are stored in {run_name}.pickle.')
    parser.add_argument('--master_seed', default = 42, type = int, help = 'Master seed for generating identical runs. For a fixed tuple (n_runs, master_seed), the runs are always the same regardless of the method.')
    parser.add_argument('--prompts_types', default = 'standard', type = str, help = 'Type of prompts. One of \' standard \', ') #TODO
    parser.add_argument('--prompts_n_shots', default = None, type = int)
    parser.add_argument('--prompts_seed', default = None, type = int)

    # Parse initial arguments
    args, remaining_args = parser.parse_known_args()

    # Conditionally add named arguments for OGA
    if args.adapt_method_name == 'OGA':
        oga_parser = argparse.ArgumentParser()
        oga_parser.add_argument('--tau', default = 0.05, type=float, help="tau parameter of the MAP estimator.")
        oga_parser.add_argument('--sig_type', default = 'RidgeMoorePenrose',type=str, help="Type of the estimator of inverse matrix. One of Ridge (always use Bayes-Ridge estimator) , MoorePenrose (always use (pseudo-)inverse) or RidgeMoorePenrose (switch estimator based on the number of samples available for estimation). ")
        oga_parser.add_argument('--normalize_mu', default = False, type=bool, help="Wether to normalize mu after estimation.")
        oga_args = oga_parser.parse_args(remaining_args)
        
        # Add parsed OGA-specific parameters to the main args
        # for key, value in vars(oga_args).items():
        #     setattr(args, key, value)
        args.OGA_params = oga_args
        print(args)
    elif args.adapt_method_name == 'TDA':
        tda_parser = argparse.ArgumentParser()
        tda_parser.add_argument('--pos_cache_logits_scale', default = 2.0, type = float, help = "Scaling factor for logits obtained with the positive cache.")
        tda_parser.add_argument('--neg_cache_logits_scale', default = 0.117, type = float, help = 'Scaling factor for logits obtained with the negative cache.')
        tda_parser.add_argument('--neg_cache_capacity', default = 3, type = int, help = 'Maximum size of the negative cache in number of shots per class. The size of the positive cache is computed to reach the target shot_capacity.')
        
        tda_args = tda_parser.parse_args(remaining_args)
        args.TDA_params = tda_args
        
    elif args.adapt_method_name == 'DMN':
        dmn_parser = argparse.ArgumentParser()
        dmn_parser.add_argument('--DMN_prob_factor', default = 1.0, type = float)
        dmn_args = dmn_parser.parse_args(remaining_args)
        args.DMN_params = dmn_args
        
    return args


def main():
    args = get_arguments()
    assert args.data_root_path is not None
    runner.run(args)
    return None


if __name__ == '__main__':
    main()