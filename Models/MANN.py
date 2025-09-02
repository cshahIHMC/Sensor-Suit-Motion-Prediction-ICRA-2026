import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class Model(torch.nn.Module):
    # def __init__(self, gating_indices, gating_input, gating_hidden, gating_output, main_indices, main_input, main_hidden, main_output, dropout, input_norm, output_norm):
    def __init__(self, gating_input, gating_hidden, gating_output, main_input, main_hidden, main_output, dropout):
        super(Model, self).__init__()

        # if len(gating_indices) + len(main_indices) != len(input_norm[0]):
        #     print("Warning: Number of gating features (" + str(len(gating_indices)) + ") and main features (" + str(len(main_indices)) + ") are not the same as input features (" + str(len(input_norm[0])) + ").")

        self.G1 = nn.Linear(gating_input, gating_hidden)
        self.G2 = nn.Linear(gating_hidden, gating_hidden)
        self.G3 = nn.Linear(gating_hidden, gating_output)

        self.E1 = ExpertLinear(gating_output, main_input, main_hidden)
        self.E2 = ExpertLinear(gating_output, main_hidden, main_hidden)
        self.E3 = ExpertLinear(gating_output, main_hidden, main_hidden)
        self.E4 = ExpertLinear(gating_output, main_hidden, main_output)
        
        fab_mean = [2.377953463,	1.849263019, 1.60007859, 1.757481385,	2.888998,	1.900023537,	3.556215491,	2.612263944,	1.895851205,	4.202264476,
        24.57707941,	35.45750622,	30.4602287,	28.32753799,	20.19461924,	30.86888558,	15.07308212,	21.87154258,	22.54799156,	47.56942489,
        -0.3509646938,	0.3130965961,	0.8534762274,	-1.408647667,	0.1639256654,	0.6967448767,	-0.4100150165,	-0.1157595488,	0.3081485786,	-0.459047836  ]
        
     
        fab_std = [0.3903993294,	0.4026807299,	0.3789195593,	0.7364284741,	0.4014533982,	0.3851321317,	0.5857557042,	0.3110404101,	0.4520797378,	0.4512631504,
        13.05412009,	20.90036521,	15.25565521,	16.41055256,	9.506750442,	13.3120746,	7.822642518,	10.1290902,	12.29267163,	23.88696694,
        9.140419992,	9.249584712,	7.384249101,	10.3740144,	7.440053835,	11.04651876,	6.266527292,	7.450460056,	6.397610661,	5.641210812]

        # Register as buffers so they move with .to(device) / .cuda()
        self.register_buffer("fab_mean", torch.tensor(fab_mean, dtype=torch.float32))
        self.register_buffer("fab_std", torch.tensor(fab_std, dtype=torch.float32))


        self.dropout = dropout
        # self.Xnorm = Parameter(torch.from_numpy(input_norm), requires_grad=False)
        # self.Ynorm = Parameter(torch.from_numpy(output_norm), requires_grad=False)

    def forward(self, phaseinputs, motionpredictionInputs):
        # x = utility.Normalize(x, self.Xnorm)
        
        

        #Gating
        g = phaseinputs
        g[:, 20:50] = (g[:, 20:50] - self.fab_mean) / (self.fab_std + 1e-6)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G1(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G2(g)
        g = F.elu(g)

        g = F.dropout(g, self.dropout, training=self.training)
        g = self.G3(g)

        w = F.softmax(g, dim=1)

        #Main
        m = motionpredictionInputs

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E1(m, w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E2(m , w)
        m = F.elu(m)

        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E3(m, w)
        m = F.elu(m)
        
        m = F.dropout(m, self.dropout, training=self.training)
        m = self.E4(m, w)
    
   
        return m, w

#Output-Blended MoE Layer
class ExpertLinear(torch.nn.Module):
    def __init__(self, experts, input_dim, output_dim):
        super(ExpertLinear, self).__init__()

        self.experts = experts
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = self.weights([experts, input_dim, output_dim])
        self.b = self.bias([experts, 1, output_dim])

    def forward(self, x, weights):
        y = torch.zeros((x.shape[0], self.output_dim), device=x.device, requires_grad=True)
        for i in range(self.experts):
            y = y + weights[:,i].unsqueeze(1) * (x.matmul(self.W[i,:,:]) + self.b[i,:,:])
        return y

    def weights(self, shape):
        alpha_bound = np.sqrt(6.0 / np.prod(shape[-2:]))
        alpha = np.asarray(np.random.uniform(low=-alpha_bound, high=alpha_bound, size=shape), dtype=np.float32)
        return Parameter(torch.from_numpy(alpha), requires_grad=True)

    def bias(self, shape):
        return Parameter(torch.zeros(shape, dtype=torch.float), requires_grad=True)
