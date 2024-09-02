import argparse
import os.path as p

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='cpu if running on CPU or cuda if running on GPU')
parser.add_argument('--data_filename', type=str, default=p.join('data', 'NDW', 'speed_info.h5'), help='data path')
parser.add_argument('--adj_filename', type=str, default=p.join('data', 'NDW', 'rotterdam_ring_adjacency.xlsx'),
                    help='adj data path')
parser.add_argument('--num_of_weeks', type=int, default=5, help='number of previous weeks for temporal pattern')
parser.add_argument('--num_of_days', type=int, default=6, help='number of previous days for temporal pattern')
parser.add_argument('--num_of_hours', type=int, default=4, help='number of previous hours for temporal pattern')
parser.add_argument('--data_per_hour', type=int, default=60, help='The frequency of recording time series data per hour')
parser.add_argument('--num_of_vertices', type=int, default=206, help='vertices in the transportation graph')
parser.add_argument('--artificial_missing_ratio', type=int, default=25, help='The percentage of the observed values in'
                                                                             'each sample take as artificial missing '
                                                                             'values')
parser.add_argument('--k', type=int, default=2, help='order of chebyshev polynomials')
parser.add_argument('--batch_size', type=int, default=24, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')

parser.add_argument('--save', type=str, default=p.join('data', 'NDW','val'), help='save model')
parser.add_argument('--save_outputs', type=str, default= p.join('data', 'NDW','output_values.xlsx'),
                    help='save est error for each segment, test set')

parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--fold', type=int, default=5, help='number of folds')
parser.add_argument('--kernel_num', type=int, default=32, help='Size of hidden units')
