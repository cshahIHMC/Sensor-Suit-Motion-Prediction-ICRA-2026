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
        
        fab_mean = [-3.2564923e-02, -2.3406364e-02, 1.3240871e+00, 4.1014603e+01,
                    -2.4297564e-01, -1.6370684e-02, -2.4113677e-02, 1.7107403e+00,
                     5.4285000e+01,  4.5630690e-02,  4.6314280e-02,  1.9869128e-02,
                     4.6051321e+00,  1.6846336e+01, -4.2095739e-01, -4.5052882e-02,
                     1.0535752e-01,  3.1186976e+00,  2.3081659e+01, -2.0144255e-01,
                     3.3257127e-02,  2.9932617e-03,  1.4969879e+00,  3.5440731e+01,
                    -1.5683125e-01, -4.0017970e-02, -4.4576611e-02,  2.8906257e+00,
                     2.4773144e+01,  1.8246192e+00]

        
     
        fab_std = [ 0.7077086,   0.70508003,  0.3104338,  16.11017   ,  6.2945657 ,  0.66618043,
                    0.7445238,   0.34715655, 15.923401 ,   6.063388  ,  0.705116  ,  0.70655215,
                    1.9695925,  10.840973  ,  3.1709101,   0.6837597 ,  0.72021097,  2.0887349,
                   14.667681 ,   3.639706  ,  0.7222788,   0.69070977,  0.38787553, 16.480913,
                    5.2354403,   0.7105775 ,  0.7003866,   0.99095803, 14.494578  ,  7.0202174 ]
                          
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
