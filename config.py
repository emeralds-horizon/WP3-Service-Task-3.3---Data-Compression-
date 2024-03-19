import argparse
import os.path as p

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='cpu if running on CPU or cuda if running on GPU')
parser.add_argument('--data_filename', type=str, default=p.join('data', 'PEMS04', 'pems04.npz'), help='data path')
parser.add_argument('--adj_filename', type=str, default=p.join('data', 'PEMS04', 'distance.csv'),
                    help='adj data path')
parser.add_argument('--num_of_weeks', type=int, default=5, help='number of previous weeks for temporal pattern')
parser.add_argument('--num_of_days', type=int, default=6, help='number of previous days for temporal pattern')
parser.add_argument('--num_of_hours', type=int, default=4, help='number of previous hours for temporal pattern')
parser.add_argument('--points_per_hour', type=int, default=12, help='only used for PEMS data (?)')
parser.add_argument('--num_of_vertices', type=int, default=307, help='vertices in the transportation graph')
parser.add_argument('--k', type=int, default=2, help='order of chebyshev polynomials')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')


parser.add_argument('--save', type=str, default=p.join('val', 'chengdu40'), help='save path')
parser.add_argument('--save_errors', type=str, default='output_values.xlsx', help='save pred error for each segment')


parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--fold', type=int, default=5, help='number of folds')
parser.add_argument('--kernel_num', type=int, default=32, help='Size of hidden units')
