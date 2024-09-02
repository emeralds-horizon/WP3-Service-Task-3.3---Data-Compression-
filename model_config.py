from data_prepration import scaled_laplacian_matrix, adjacency_matrix, cheb_polynomial
import torch


def set_backbones(adj_filename, num_of_vertices, num_of_weeks, num_of_days, num_of_hours, k, roads_names):
    adj_mx = adjacency_matrix(adj_filename, num_of_vertices, roads_names)  # shape: num_of_vertices * num_of_vertices
    lp_mx = scaled_laplacian_matrix(adj_mx)  # shape: num_of_vertices * num_of_vertices
    cheb_polynomials = [torch.tensor(i, device='cpu') for i in cheb_polynomial(lp_mx, k)]
    # shape: k * num_of_vertices * num_of_vertices

    backbones_weeks = [
        {
            "K": k,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "num_of_input_channels": 1,
            "time_conv_strides": num_of_weeks,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": k,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "num_of_input_channels": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones_days = [
        {
            "K": k,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "num_of_input_channels": 1,
            "time_conv_strides": num_of_days,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": k,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "num_of_input_channels": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones_hours = [
        {
            "K": k,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "num_of_input_channels": 1,
            "time_conv_strides": num_of_hours,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": k,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "num_of_input_channels": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    all_backbones = [
        backbones_weeks,
        backbones_days,
        backbones_hours
    ]

    return ['week', 'day', 'hour'], all_backbones

