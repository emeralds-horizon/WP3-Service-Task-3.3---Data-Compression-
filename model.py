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

        self.U_1 = nn.Parameter(torch.Tensor())
        self.U_2 = nn.Parameter(torch.Tensor())
        self.U_3 = nn.Parameter(torch.Tensor())
        self.b_e = nn.Parameter(torch.Tensor())
        self.V_e = nn.Parameter(torch.Tensor())

        self.init_scale = 0.1

    def initialize_parameters(self, num_of_vertices, num_of_features, num_of_timesteps):
        self.U_1.data = F.pad(self.init_scale * torch.randn(num_of_vertices).double(), (0, 0))
        self.U_2.data = F.pad(self.init_scale * torch.randn(num_of_features, num_of_vertices).double(), (0, 0))
        self.U_3.data = F.pad(self.init_scale * torch.randn(num_of_features).double(), (0, 0))
        self.b_e.data = F.pad(self.init_scale * torch.randn(1, num_of_timesteps, num_of_timesteps).double(), (0, 0))
        self.V_e.data = F.pad(self.init_scale * torch.randn(num_of_timesteps, num_of_timesteps).double(), (0, 0))

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor, shape is (batch_size, N, V, T)

        Returns
        ----------
        e_normalized: torch tensor, temporal attention scores shape is (batch_size, T, T)
        """

        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Lazy initialization
        if not hasattr(self, 'U_1') or not hasattr(self.U_1, 'data') or self.U_1.data.numel() == 0:
            self.initialize_parameters(num_of_vertices, num_of_features, num_of_timesteps)

        # compute temporal attention scores > shape: [Batch_size, Time, Vertices]
        tmp1 = torch.matmul(x.permute(0, 3, 2, 1).double(), self.U_1.data.double()).double()
        lhs = torch.matmul(tmp1, self.U_2.data.double())
        # shape: [Batch_size, Vertices, Time]
        rhs = torch.matmul(x.permute(0, 1, 3, 2).double(), self.U_3.data.double())
        # shape: [Batch_size, Time, Time]
        product = torch.bmm(lhs, rhs)

        sigmoid_product = torch.sigmoid(product + self.b_e.data.double())
        e = torch.matmul(self.V_e.data.double(), sigmoid_product.permute(1, 2, 0)).permute(2, 0, 1)

        # normalization
        e = e - torch.max(e, dim=1, keepdim=True)[0]
        exp_e = torch.exp(e)
        e_normalized = exp_e / torch.sum(exp_e, dim=1, keepdim=True)

        return e_normalized  # shape: [Batch_size, Time, Time]


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

    def initialize_parameters(self, num_of_features):
        self.Theta.data = F.pad(self.init_scale * torch.randn(self.K, num_of_features, self.num_of_filters).double(), (0, 0))

    def forward(self, x, spatial_attention):
        """
        Chebyshev graph convolution operation

        Parameters
        ----------
        x: torch tensor, shape is (batch_size, N, V, T)

        spatial_attention: torch tensor, shape is (batch_size, Vertices, Vertices)
                           spatial attention scores

        Returns
        ----------
        mx.ndarray, shape is (batch_size, N, self.num_of_filters, T_{r-1})

        """
        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape

        # Lazy initialization
        if not hasattr(self, 'Theta') or not hasattr(self.Theta, 'data') or self.Theta.data.numel() == 0:
            self.initialize_parameters(num_of_features)

        outputs = []
        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]

            # shape: [batch_size, V, F]
            output = torch.zeros((batch_size, num_of_vertices, self.num_of_filters), device=x.device)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # shape: [V, V]
                T_k_with_at = T_k * spatial_attention  # shape: [batch_size, V, V]
                theta_k = self.Theta.data[k].clone()  # shape: [number_of_features, num_of_filters]
                rhs = torch.bmm(T_k_with_at.permute(0, 2, 1).double(),
                                graph_signal.double()) # shape: [batch_size, Vertices, Features]

                output = output + torch.matmul(rhs.double(), theta_k.double())

            outputs.append(output.unsqueeze(-1))

        return torch.relu(torch.cat(outputs, dim=-1))  # shape: [Batches, Vertices, Filters, Times]


class SpatialAttentionLayer(nn.Module):
    """
    compute spatial attention scores
    """
    def __init__(self):
        super(SpatialAttentionLayer, self).__init__()

        self.W_1 = nn.Parameter(torch.Tensor())
        self.W_2 = nn.Parameter(torch.Tensor())
        self.W_3 = nn.Parameter(torch.Tensor())
        self.b_s = nn.Parameter(torch.Tensor())
        self.V_s = nn.Parameter(torch.Tensor())
        self.init_scale = 0.1

    def initialize_parameters(self, num_of_vertices, num_of_features, num_of_timesteps):
        self.W_1.data = F.pad(self.init_scale * torch.randn(num_of_timesteps).double(), (0, 0))
        self.W_2.data = F.pad(self.init_scale * torch.randn(num_of_features, num_of_timesteps).double(), (0, 0))
        self.W_3.data = F.pad(self.init_scale * torch.randn(num_of_features).double(), (0, 0))
        self.b_s.data = F.pad(self.init_scale * torch.randn(1, num_of_vertices, num_of_vertices).double(), (0, 0))
        self.V_s.data = F.pad(self.init_scale * torch.randn(num_of_vertices, num_of_vertices).double(), (0, 0))

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor, shape is (batch_size, N, V, T)

        Returns
        ----------
        S_normalized: torch tensor, spatial attention scores shape is (batch_size, N, N)
        """

        # get shape of input matrix x
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # Lazy initialization
        if not hasattr(self, 'W_1') or not hasattr(self.W_1, 'data') or self.W_1.data.numel() == 0:
            self.initialize_parameters(num_of_vertices, num_of_features, num_of_timesteps)

        # compute spatial attention scores
        lhs = torch.matmul(torch.matmul(x, self.W_1.data.double()), self.W_2.data.double())  # shape: [batch_size, V, T]
        rhs = torch.matmul(x.permute(0, 3, 1, 2).double(), self.W_3.data.double())  # shape: [batch_size, T, V]
        product = torch.bmm(lhs, rhs)  # shape: [batch_size, V, V]

        # S computation
        sigmoid_s = torch.sigmoid(product + self.b_s.data.double())
        s = torch.matmul(self.V_s.data.double(), sigmoid_s.permute(1, 2, 0)).permute(2, 0, 1)

        # normalization
        s = s - torch.max(s, dim=1, keepdim=True)[0]
        exp_s = torch.exp(s)
        s_normalized = exp_s / torch.sum(exp_s, dim=1, keepdim=True)

        return s_normalized


class GCNBlock(nn.Module):
    """
    Each submodule contains one or more GCN block,
    based on its backbone in model_config
    """
    def __init__(self, backbone):
        """
        Parameters
        ----------
        backbone: dict, have 5 keys,
                        "K",
                        "num_of_chev_filters",
                        "num_of_time_filters",
                        "time_conv_strides",
                        "cheb_polynomials"
        """

        super(GCNBlock, self).__init__()

        K = backbone['K']
        num_of_chev_filters = backbone['num_of_chev_filters']
        num_of_time_filters = backbone['num_of_time_filters']
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
            stride=(1, 1))
        self.residual_conv1 = nn.Conv2d(  # once 3 and once 64?
            in_channels=3,  # ###################
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, 1))

        self.residual_conv2 = nn.Conv2d(
            in_channels=64,  # ###################
            out_channels=num_of_time_filters,
            kernel_size=(1, 1),
            stride=(1, 1))

        # self.ln = nn.LayerNorm(axis=2)  ???? from MXnet in this line, change it to the next line
        self.in_channels = 64
        # do this based on sub_net name ??????
        self.ln_w = nn.LayerNorm(normalized_shape=torch.Size([args.batch_size, args.num_of_vertices, self.in_channels,
                                                              args.num_of_weeks]), elementwise_affine=True)
        self.ln_d = nn.LayerNorm(normalized_shape=torch.Size([args.batch_size, args.num_of_vertices, self.in_channels,
                                                              args.num_of_days]), elementwise_affine=True)
        self.ln_h = nn.LayerNorm(normalized_shape=torch.Size([args.batch_size, args.num_of_vertices, self.in_channels,
                                                              args.num_of_hours]), elementwise_affine=True)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch tensor, batch_size, num_of_vertices, num_of_features, num_of_timesteps

        x: torch tensor, shape is (batch_size, N, C_{r-1}, T_{r-1}) ????

        Returns
        ----------
        ndarray, shape is (batch_size, N, num_of_time_filters, T_{r-1})

        """

        (batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape

        temporal_at = self.TAt(x)  # shape: [batch_size, T, T]
        # shape: [batch_size, V, Features, T]
        x_tat = torch.bmm(x.view(batch_size, -1, num_of_timesteps).double(), temporal_at.double()).\
            view(batch_size, num_of_vertices, num_of_features, num_of_timesteps)

        # cheb gcn with spatial attention
        spatial_at = self.SAt(x_tat)
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at)

        # convolution along the time axis
        time_conv_output = self.time_conv(spatial_gcn.permute(0, 2, 1, 3).float()).\
            permute((0, 2, 1, 3))

        # residual:  # set it auto, based on the number of ST blocks
        if num_of_features == 3:
            x_residual = self.residual_conv1(x.permute(0, 2, 1, 3).float()).permute(0, 2, 1, 3)
        elif num_of_features == self.in_channels:
            x_residual = self.residual_conv2(x.permute(0, 2, 1, 3).float()).permute(0, 2, 1, 3)
        else:
            print('Error in residual')
            exit()

        # relu:
        if num_of_timesteps == args.num_of_weeks:
            rsl = self.ln_w(torch.relu(x_residual + time_conv_output))
        elif num_of_timesteps == args.num_of_days:
            rsl = self.ln_d(torch.relu(x_residual + time_conv_output))
        elif num_of_timesteps == args.num_of_hours:
            rsl = self.ln_h(torch.relu(x_residual + time_conv_output))
        else:
            print('Error in normalization layer')
            exit()

        return rsl


class GCNSubmodule(nn.Module):
    """
        a module in GCN: 1. week, 2.day, 3. week
    """

    def __init__(self, backbones):
        """
            Parameters
            ----------
            backbones: list(dict), list of backbones for the current submodule

        """

        super(GCNSubmodule, self).__init__()

        # For each backbone, create an instance of the GCNBlock class
        self.blocks = nn.Sequential()
        for backbone in backbones:
            self.blocks.add_module('astgcn_block', GCNBlock(backbone))
        self.W = nn.Parameter(torch.Tensor())
        # final fully connected layer
        self.final_conv_w = nn.Conv2d(in_channels=args.num_of_weeks,
                                      out_channels=1,
                                      kernel_size=(1, backbones[-1]['num_of_time_filters']))
        self.final_conv_d = nn.Conv2d(in_channels=args.num_of_days,
                                      out_channels=1,
                                      kernel_size=(1, backbones[-1]['num_of_time_filters']))
        self.final_conv_h = nn.Conv2d(in_channels=args.num_of_hours,
                                      out_channels=1,
                                      kernel_size=(1, backbones[-1]['num_of_time_filters']))

    def initialize_parameters(self, num_of_vertices, num_for_prediction):
        self.W.data = F.relu(torch.randn(num_of_vertices, num_for_prediction).double(), (0, 0))

    def forward(self, x):
        """
            Parameters
            ----------
            x: torch.Tensor, shape is (batch_size, num_of_vertices, num_of_features, num_of_timesteps)

            Returns
            ----------
            torch.ndarray, shape is (batch_size, num_of_vertices)????
        """

        x = self.blocks(x)

        # hard code!!!!!!!!!!!!!!!!!!
        # instead of checking timesteps, check sub_name, since we could have same timesteps
        # final convolution + Relu

        timestep = x.shape[3]
        if timestep == args.num_of_weeks:  # week
            module_output = torch.relu(self.final_conv_w(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1))

        elif timestep == args.num_of_days:  # day
            module_output = torch.relu(self.final_conv_d(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1))

        elif timestep == args.num_of_hours:  # hour
            module_output = torch.relu(self.final_conv_h(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1))


        _, num_of_vertices, num_for_prediction = module_output.shape

        # Lazy initialization
        if not hasattr(self, 'W') or not hasattr(self.W, 'data') or self.W.data.numel() == 0:
            _, num_of_vertices, num_for_prediction = module_output.shape
            self.initialize_parameters(num_of_vertices, num_for_prediction)

        return module_output * self.W.data


class GCN(nn.Module):
    """
    ASTGCN, 3 sub-modules, for hour, day, week respectively
    """

    def __init__(self, sub_net_name, all_backbones):
        """
        Parameters
        ----------
        all_backbones: list[list], 3 backbones for "week", "day", "hour" submodules
        sub_net_name: list[list], 3 string names: week, day, hour
        """

        super(GCN, self).__init__()
        if len(all_backbones) <= 0:
            raise ValueError("The length of all_backbones must be greater than 0")

        self.submodules = nn.ModuleList()

        for i in range(len(all_backbones)):
            self.submodules.append(GCNSubmodule(all_backbones[i]))
            self.add_module(sub_net_name[i], self.submodules[-1])

    def forward(self, data):
        """
        Parameters
        ----------
        data: list[torch.ndarray], including week, day, recent
            each section shape is (batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        """

        if len(data) != len(self.submodules):
            raise ValueError("num of submodule not equals to length of the input list")

        num_of_vertices_set = {i.shape[1] for i in data}
        if len(num_of_vertices_set) != 1:
            raise ValueError("Different num_of_vertices detected! Check if your input data have same size at axis 1.")

        batch_size_set = {i.shape[0] for i in data}
        if len(batch_size_set) != 1:
            raise ValueError("Input values must have same batch size!")

        submodule_outputs = [self.submodules[idx](data[idx]) for idx in range(len(self.submodules))]

        return torch.sum(torch.stack(submodule_outputs), dim=0)


class Optimizer:
    def __init__(self, sub_net_name, all_backbones):
        """
        sub_net_name: list[list], 3 string names: week, day, hour
        all_backbones: list[list], 3 backbones for "week", "day", "hour" submodules
        device: str, cpu or gpu
        """

        self.model = GCN(sub_net_name, all_backbones)
        self.model.to(args.device)
        # self.initialization = Initialization()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
        # self.loss = util.masked_mae  # less sensitive to outliers
        self.loss = nn.MSELoss()
        self.clip = 5

    def train(self, data, real):
        """
        :param data: list[torch.Tensor], including: [train_week, Train_day, Train_recent]
        :param real: torch.Tensor, including current time real values
        :return: ???
        """

        self.model.train()  # sets the model in training mode
        self.optimizer.zero_grad()  # initializes the gradients
        # maybe using initialization wrote in the first lines of this page ... ???

        logdir = '/logs'  # Replace with your desired path
        logdir = tempfile.mkdtemp()
        # Initialize SummaryWriter
        # sw = SummaryWriter(logdir=logdir, flush_secs=5)

        # training:
        output = self.model(data)

        # real: torch.ndarray, shape is (batch_size, num_of_vertices)  -> including the predicted value
        # ??? put a if here, and do this line just when we have 1 time window for prediction
        t_real = real.unsqueeze(2)

        loss = self.loss(output, t_real)
        loss.backward()  # computes the gradients of the loss using back-propagation

        # updating parameters:
        if self.clip is not None:  # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()  # updates the model parameters

        mape = metrics.masked_mape_np(output, t_real, 0.0).item()
        rmse = metrics.mean_squared_error(output, t_real).item()

        return [loss.item(), mape, rmse]

    def eval(self, data, real):
        self.model.eval()
        output = self.model(data)

        t_real = real.unsqueeze(2)

        loss = self.loss(output, t_real)

        return loss.item()
