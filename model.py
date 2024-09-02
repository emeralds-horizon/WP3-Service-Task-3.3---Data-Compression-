import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
# from tensorboardX import SummaryWriter
import tempfile
import utils
import metrics
from config import parser

args = parser.parse_args()


class Initialization(nn.Module):
    def __init__(self):
        super(Initialization, self).__init__()

    @staticmethod
    def init_weight(model):
        if len(model.shape) < 2:
            init.uniform_(model)
            print(f'Init {model.shape} with Uniform')
        else:
            init.xavier_uniform_(model)
            print(f'Init {model.shape} with Xavier')


class TemporalAttentionLayer(nn.Module):
    """
    compute temporal attention scores
    """
    def __init__(self):
        super(TemporalAttentionLayer, self).__init__()

        self.U_1 = nn.Parameter(torch.Tensor())  # transform the input in the spatial dimension
        self.U_2 = nn.Parameter(torch.Tensor())  # combine features across time steps
        self.U_3 = nn.Parameter(torch.Tensor())  # aggregate the temporal features
        self.b_e = nn.Parameter(torch.Tensor())  # bias
        self.W_e = nn.Parameter(torch.Tensor())  # learnable weight matrix

        self.init_scale = 0.1
        self.initialized = False

    def initialize_parameters(self, num_of_vertices, num_of_features, num_of_timesteps):
        # self.U_1.data = F.pad(self.init_scale * torch.randn(num_of_vertices).double(), (0, 0))
        # self.U_2.data = F.pad(self.init_scale * torch.randn(num_of_features, num_of_vertices).double(), (0, 0))
        # self.U_3.data = F.pad(self.init_scale * torch.randn(num_of_features).double(), (0, 0))
        # self.b_e.data = F.pad(self.init_scale * torch.randn(1, num_of_timesteps, num_of_timesteps).double(), (0, 0))
        # self.V_e.data = F.pad(self.init_scale * torch.randn(num_of_timesteps, num_of_timesteps).double(), (0, 0))
        # self.init_parameters()
        self.U_1 = nn.Parameter(self.init_scale * torch.randn(num_of_vertices).double())
        self.U_2 = nn.Parameter(self.init_scale * torch.randn(num_of_features,num_of_vertices).double())
        self.U_3 = nn.Parameter(self.init_scale * torch.randn(num_of_features).double())
        self.b_e = nn.Parameter(self.init_scale * torch.randn(1, num_of_timesteps, num_of_timesteps).double())
        self.W_e = nn.Parameter(self.init_scale * torch.randn(num_of_timesteps, num_of_timesteps).double())
        self.initialized = True

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor, shape is (batch_size, V, F, T)

        Returns
        ----------
        e_normalized: torch tensor, temporal attention scores shape is (batch_size, T, T)
        """

        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Lazy initialization
        if not self.initialized:
            self.initialize_parameters(num_of_vertices, num_of_features, num_of_timesteps)

        # (X^{(r-1)})^T U_1 U_2
        X_T = x.permute(0, 3, 2, 1)  # shape: (batch_size, num_of_timesteps, num_of_features, num_of_vertices)

        term1 = torch.matmul(X_T, self.U_1)  # shape: (batch_size, num_of_timesteps, num_of_features)
        term1 = torch.matmul(term1, self.U_2)  # shape: (batch_size, num_of_timesteps, num_of_vertices)

        # U_3 X^{(r-1)}
        # shape: (batch_size, num_of_vertices, num_of_timesteps)
        if len(self.U_3) != 1: 
            # term2 = torch.matmul(self.U_3, x.permute(2, 0, 1, 3))
            term2 = torch.einsum('i,ijkl->jkl', self.U_3, x.permute(2, 0, 1, 3))
        else:
            term2 = x.squeeze(2)

        # combine the terms
        product = torch.matmul(term1, term2)  # shape: (batch_size, num_of_timesteps, num_of_timesteps)

        # compute attention scores
        E = torch.sigmoid(product + self.b_e)  # shape: (batch_size, num_of_timesteps, num_of_timesteps)

        # apply the final transformation V_e
        E = torch.matmul(E, self.W_e)  # shape: (batch_size, num_of_timesteps, num_of_timesteps)

        # normalization
        # apply softmax to get attention weights
        E_normalized = F.softmax(E, dim=-1)  # shape: (batch_size, num_of_timesteps, num_of_timesteps)

        return E_normalized.float()  # shape: [Batch_size, Time, Time]


class ChebConvWithSAt(nn.Module):
    """
    K-order chebyshev graph convolution with Spatial Attention scores
    """
    def __init__(self, num_of_filters, K, cheb_polynomials):
        """
        Parameters
        ----------
        num_of_filters: int

        num_of_features: int, num of input features

        K: int, up K - 1 order chebyshev polynomials
                will be used in this convolution

        cheb_polynomials: list[torch tensor], length: K, from T_0 to T_{K-1}

        """

        super(ChebConvWithSAt, self).__init__()
        self.K = K
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = cheb_polynomials
        self.Theta = nn.Parameter(torch.Tensor())
        self.init_scale = 0.1
        self.initialized = False

    def initialize_parameters(self, num_of_features):
        self.Theta = nn.Parameter(self.init_scale * torch.randn(self.K, num_of_features, self.num_of_filters).double())

        self.initialized = True

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: torch tensor, shape is (batch_size, num_of_timesteps, num_of_features, num_of_vertices)

        spatial_attention: torch tensor, shape is (batch_size, Vertices, Vertices)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, num_of_vertices, self.num_of_filters, T_{r-1})

        """

        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Lazy initialization
        if not self.initialized:
            self.initialize_parameters(num_of_features)  

        outputs = []
        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # shape: (batch_size, num_of_features, num_of_vertices)

            output = torch.zeros((batch_size, num_of_vertices, self.num_of_filters), device=x.device)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # shape: (num_of_vertices, num_of_vertices)
                T_k_with_at = T_k * spatial_attention  # shape: (batch_size, num_of_vertices, num_of_vertices)
                theta_k = self.Theta.data[k].clone()  # shape: (num_of_features, num_of_filters)

                # apply Chebyshev convolution
                rhs = torch.bmm(T_k_with_at.permute(0, 2, 1), graph_signal)   # shape: (batch_size, num_of_vertices, num_of_features)
                tmp = torch.matmul(rhs, theta_k.unsqueeze(0))  # shape: (batch_size, num_of_vertices, num_of_filters)
                output = output + tmp  # shape: (batch_size, num_of_vertices, num_of_filters)

            outputs.append(output.unsqueeze(-1))  # shape: (batch_size, num_of_vertices, num_of_filters, 1)

        return torch.relu(torch.cat(outputs, dim=-1))   # shape: (batch_size, num_of_vertices, num_of_filters, num_of_timesteps)


class SpatialAttentionLayer(nn.Module):
    """
    compute spatial attention scores
    """
    def __init__(self):
        super(SpatialAttentionLayer, self).__init__()

        self.W_1 = nn.Parameter(torch.Tensor())  # transform the input in the time dimension
        self.W_2 = nn.Parameter(torch.Tensor())  # combine features across vertices
        self.W_3 = nn.Parameter(torch.Tensor())  # transform the transposed input tensor
        self.b_s = nn.Parameter(torch.Tensor())  # bias
        self.W_s = nn.Parameter(torch.Tensor())  # learnable weight matrix
        self.init_scale = 0.1
        self.initialized = False

    def initialize_parameters(self, num_of_vertices, num_of_features, num_of_timesteps):
        self.W_1 = nn.Parameter(self.init_scale * torch.randn(num_of_timesteps).double())
        self.W_2 = nn.Parameter(self.init_scale * torch.randn(num_of_features, num_of_timesteps).double())
        self.W_3 = nn.Parameter(self.init_scale * torch.randn(num_of_features).double())
        self.b_s = nn.Parameter(self.init_scale * torch.randn(1, num_of_vertices, num_of_vertices).double())
        self.W_s = nn.Parameter(self.init_scale * torch.randn(num_of_vertices, num_of_vertices).double())

        self.initialized = True

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor, shape is (batch_size, V, F, T)

        Returns
        ----------
        S_normalized: torch tensor, spatial attention scores shape is (batch_size, N, N)
        """

        # get shape of input matrix x
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Lazy initialization
        if not self.initialized:
            self.initialize_parameters(num_of_vertices, num_of_features, num_of_timesteps)

        # compute spatial attention scores

        # (X W_1) W_2
        # shape: (batch_size, num_of_vertices, num_of_features)
        term1 = torch.matmul(x, self.W_1)
        term1 = torch.matmul(term1, self.W_2)

        # W_3 X^T
        # shape: (batch_size, num_of_timesteps, num_of_vertices)
        if len(self.W_3) != 1:  
            # term2 = torch.matmul(self.W_3, x.permute(2, 0, 3, 1))
            term2 = torch.einsum('i,ijkl->jkl', self.W_3, x.permute(2, 0, 3, 1))
        else:
            term2 = x.permute(0, 3, 2, 1).squeeze(2)

        # combine the terms
        product = torch.matmul(term1, term2)  # shape: (batch_size, num_of_vertices, num_of_vertices)

        # compute attention scores
        S = torch.sigmoid(product + self.b_s)  # shape: (batch_size, num_of_vertices, num_of_vertices)

        # apply the final transformation V_s
        S = torch.matmul(S, self.W_s)  # shape: (batch_size, num_of_vertices, num_of_vertices)

        # apply softmax to get normalized attention scores
        S_normalized = F.softmax(S, dim=-1)  # shape: (batch_size, num_of_vertices, num_of_vertices)

        return S_normalized.float()  # shape: [batch_size, num_of_vertices, num_of_vertices]


class GCNBlock(nn.Module):
    """
    Each submodule contains one or more GCN block,
    based on its backbone in model_config
    """
    def __init__(self, backbone, sub_net_name):
        """
        Parameters
        ----------
        backbone: dict, have 5 keys,
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_strides",
                        "cheb_polynomials"
        sub_net_name: str, one of "week", "day", "hour"
        """

        super(GCNBlock, self).__init__()

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
        num_of_input_channels = backbone['num_of_input_channels']
        time_conv_strides = backbone['time_conv_strides']
        cheb_polynomials = backbone["cheb_polynomials"]

        self.SAt = SpatialAttentionLayer()
        self.cheb_conv_SAt = ChebConvWithSAt(
            num_of_filters=num_of_chev_filters,
            K=K,
            cheb_polynomials=cheb_polynomials)
        self.TAt = TemporalAttentionLayer()
        self.time_conv = nn.Conv2d(
            in_channels=num_of_time_filters,
            out_channels=num_of_time_filters,
            kernel_size=(1, 3),
            padding=(0, 1),
            stride=(1, 1))  # strides=(1, time_conv_strides))
        self.residual_conv = nn.Conv2d(
            in_channels=num_of_input_channels,
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, 1))  # strides=(1, time_conv_strides))

        self.in_channels = 64

        if sub_net_name == "week":
            self.ln = nn.LayerNorm(
                normalized_shape=[args.batch_size, args.num_of_vertices, self.in_channels, args.num_of_weeks+1],
                elementwise_affine=True)
        elif sub_net_name == "day":
            self.ln = nn.LayerNorm(
                normalized_shape=[args.batch_size, args.num_of_vertices, self.in_channels, args.num_of_days+1],
                elementwise_affine=True)
        elif sub_net_name == "hour":
            self.ln = nn.LayerNorm(
                normalized_shape=[args.batch_size, args.num_of_vertices, self.in_channels, args.num_of_hours+1],
                elementwise_affine=True)
        else:
            raise ValueError("Invalid sub_net_name. Must be one of ['week', 'day', 'hour']")

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor, batch_size, num_of_vertices, num_of_features, num_of_timesteps

        x: torch tensor, shape is (batch_size, N, C_{r-1}, T_{r-1}) 

        Returns
        ----------
        ndarray, shape is (batch_size, N, num_of_time_filters, T_{r-1})

        """

        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape

        # temporal attention scores. shape: [batch_size, T, T]
        temporal_at = self.TAt(x)

        # apply attention scores to input. shape: [batch_size, V, T]
        x_tat = torch.bmm(x.reshape(batch_size, -1, num_of_timesteps).float(), temporal_at.float()).double()
        x_tat = x_tat.reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # cheb gcn with spatial attention
        spatial_at = self.SAt(x_tat)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at)

        # convolution along the time axis
        # shape: (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3).float()).permute((0, 2, 1, 3))

        # x_residual -> shape: (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        x_residual = self.residual_conv(x.permute(0, 2, 1, 3).float()).permute(0, 2, 1, 3)

        # rsl -> shape: (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)
        rsl = self.ln(torch.relu(x_residual + time_conv_output))

        return rsl.double()


class GCNSubmodule(nn.Module):
    """
        a submodule of GCN: 1. week, 2.day, 3. hour
    """

    def __init__(self, backbones, sub_net_name):
        """
            Parameters
            ----------
            backbones: list(dict), list of backbones for the current submodule

        """
        super(GCNSubmodule, self).__init__()

        # For each backbone of current submodule, create an instance of the GCNBlock class
        self.blocks = nn.Sequential()
        for idx, backbone in enumerate(backbones):
            self.blocks.add_module(f'tgcn_block_{idx}', GCNBlock(backbone, sub_net_name))
        self.W = nn.Parameter(torch.Tensor())

        # final fully connected layer
        if sub_net_name == "week":
            self.final_conv = nn.Conv2d(in_channels=args.num_of_weeks+1,
                                         out_channels=1,
                                         kernel_size=(1, backbones[-1]['num_of_time_filters']))
        elif sub_net_name == "day":
            self.final_conv = nn.Conv2d(in_channels=args.num_of_days+1,
                                          out_channels=1,
                                          kernel_size=(1, backbones[-1]['num_of_time_filters']))
        elif sub_net_name == "hour":
            self.final_conv = nn.Conv2d(in_channels=args.num_of_hours+1,
                                          out_channels=1,
                                          kernel_size=(1, backbones[-1]['num_of_time_filters']))
        else:
            raise ValueError("Invalid sub_net_name. Must be one of ['week', 'day', 'hour']")

        self.initialized = False

    def initialize_parameters(self, num_of_vertices):
        self.W = nn.Parameter(torch.randn(num_of_vertices))
        # self.W.data = nn.Parameter(torch.randn(num_of_vertices).double(), (0, 0))
        self.initialized = True

    def forward(self, x):
        """
            Parameters
            ----------
            x: torch.Tensor, shape is (batch_size, num_of_vertices, num_of_timesteps)

            Returns
            ----------
            torch.ndarray, shape is (batch_size, num_of_vertices)
        """
        x = x.unsqueeze(2)  # adding features: to be useful for both blocks
        x = self.blocks(x).float()  # shape: (batch_size, num_of_vertices, num_of_time_filters, num_of_timesteps)

        # final convolution + Relu  -> # shape: (batch_size, num_of_vertices)
        module_output = torch.relu(self.final_conv(x.permute(0, 3, 1, 2))[:, -1, :, -1])

        _, num_of_vertices = module_output.shape

        if not self.initialized:
            self.initialize_parameters(num_of_vertices)

        return module_output * self.W.data  # shape: (batch_size, num_of_vertices)


class GCN(nn.Module):
    """
    ASTGCN, which including 3 sub-modules, for week, day and hour respectively
    """

    def __init__(self, sub_net_names, all_backbones):
        """
        Parameters
        ----------
        sub_net_names: list[list], 3 string names: week, day, hour
        all_backbones: list[list], 3 backbones for "week", "day", "hour" submodules
        """

        super(GCN, self).__init__()
        self.sub_net_names = sub_net_names
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones must be greater than 0")

        # self.submodules = nn.ModuleList()
        self.submodules = nn.ModuleDict()

        for i in range(len(all_backbones)):
            submodule = GCNSubmodule(all_backbones[i], sub_net_names[i])
            self.submodules[sub_net_names[i]] = submodule

    def forward(self, data):
        """
        Parameters
        ----------
        data: list[torch.ndarray], including week, day, recent
            each section shape is (batch_size, num_of_vertices, num_of_timesteps)
        """

        if len(data) != len(self.submodules):
            raise ValueError("num of submodule not equals to length of the input list")

        num_of_vertices_set = {i.shape[1] for i in data}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! Check if your input data have same size at axis 1.")

        batch_size_set = {i.shape[0] for i in data}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[self.sub_net_names[idx]](data[idx])
                             for idx in range(len(self.submodules))]

        return torch.sum(torch.stack(submodule_outputs), dim=0)


class Optimizer:
    def __init__(self, sub_net_names, all_backbones):
        """
        sub_net_name: list[list], 3 string names: week, day, hour
        all_backbones: list[list], 3 backbones for "week", "day", "hour" submodules
        device: str, cpu or gpu
        """
        self.device = args.device
        self.model = GCN(sub_net_names, all_backbones)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
        self.loss = nn.MSELoss()
        self.clip = 5

        # # model summary
        # print(self.model)
        # # detailed modules
        # for name, module in self.model.named_modules():
        #     print(f"Module name: {name}, Module: {module}")
        #     print('***************************************')

    def train(self, data, current, real, data_mask, mask):
        """
        :param data: list[torch.Tensor], including: [train_week, Train_day, Train_recent]
        :param current: torch.Tensor, including current time real values  and artificial missing values
        :param real: torch.Tensor, including current time real values
        :param data_mask: list[torch.Tensor], including: [train_week_mask, Train_day_mask, Train_recent_mask] :
            including 0 and 1
        :param mask: torch.Tensor, shape: num_of_vertices. The mask indicating which values in the current data are
            observed (1), which are missing (0) or artificially missing (0.5)

        :return: errors
        """

        self.model.train()  # sets the model in training mode
        self.optimizer.zero_grad()  # initializes the gradients

        # training:
        # generator
        # concatenate current data to the historical data along the time dimension
        current = current.unsqueeze(-1)
        data = [torch.cat([d, current], dim=2) for d in data]
        output = self.model(data)  # shape: (batch_size, num_of_vertices)

        # discriminator
        output = output.float()
        real = real.float()

        # loss calculation
        edited_mask = (mask == 0.5) | (mask == 1)  # shape: (batch_size, num_of_vertices)

        loss = self.loss(output, real)
        loss = loss * edited_mask  # apply mask to loss to include observed values
        loss = loss.sum() / edited_mask.sum()  
        loss.backward()  # computes the gradients of the loss using back-propagation

        # updating parameters:
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        # error calculation
        mape = metrics.masked_mape_np(output, real, edited_mask).item()
        rmse = metrics.mean_squared_error(output, real, edited_mask).item()

        return [loss.item(), mape, rmse]


    def eval(self, data, current, real, data_mask, mask):
        """
        Evaluate the model using the given data.

        :param data: list[torch.Tensor], including: [val_week, val_day, val_recent]
        :param current: torch.Tensor, including current time real values
        :param real: torch.Tensor, including current time real values
        :param data_mask: list[torch.Tensor], including: [val_week_mask, val_day_mask, val_recent_mask]
        :param mask: torch.Tensor, indicating observed (1), missing (0), and artificially missing (0.5)

        :return: list of errors [loss, mape, rmse]
        """
        self.model.eval()
        current = current.unsqueeze(-1)
        data = [torch.cat([d, current], dim=2) for d in data]
        output = self.model(data)  # shape: (batch_size, num_of_vertices)

        output = output.float()
        real = real.float()

        # loss calculation
        edited_mask = (mask == 0.5) | (mask == 1)  # shape: (batch_size, num_of_vertices)

        loss = self.loss(output, real)
        loss = loss * edited_mask  # apply mask to loss to include observed values
        loss = loss.sum() / edited_mask.sum()  

        return loss.item()
