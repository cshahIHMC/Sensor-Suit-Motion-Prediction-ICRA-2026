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
        
        
        ########## Gyro only -  256 batch size
        
        fab_mean = [-1.3719495e-01,  1.7474059e-02,  1.3892955e+00,  3.6184311e+01,
                    -1.3428761e+00, -6.7557129e-03, -8.3445348e-02,  1.8992461e+00,
                     5.4188324e+01,  2.0814785e-01,  1.3079455e-01,  3.7251726e-02,
                     4.4906120e+00,  1.4220923e+01, -1.3069688e+00,  1.4812291e-01,
                     9.0836272e-02,  3.0402188e+00,  2.0670725e+01, -5.7229334e-01,
                     1.6584660e-01,  3.3230346e-02,  1.6489639e+00,  2.7717501e+01,
                     6.2231237e-01, -1.8604237e-01, -2.1231405e-02,  2.9361281e+00,
                     2.0367138e+01,  2.3697674e+00]
        
        
        fab_std = [ 0.7087973,   0.69134283,  0.3803319,  14.341768,    6.4755015,   0.6584229,
  0.74732214,  0.31496274, 16.881565,    5.234558,    0.6996788,   0.700879,
  1.837036,    7.2808313,   3.6568367,   0.7102944,   0.68164706,  1.8923867,
 10.547955,    3.2328355,   0.71833664,  0.6745146,   0.5018577,   12.04956,
  4.7046127,   0.70253605,   0.68597794,   0.98293644,   9.417768,    6.092425 ]


        
        ########## Gyro only -  32 batch size
        # fab_mean = [-3.2564923e-02, -2.3406364e-02, 1.3240871e+00, 4.1014603e+01,
        #             -2.4297564e-01, -1.6370684e-02, -2.4113677e-02, 1.7107403e+00,
        #              5.4285000e+01,  4.5630690e-02,  4.6314280e-02,  1.9869128e-02,
        #              4.6051321e+00,  1.6846336e+01, -4.2095739e-01, -4.5052882e-02,
        #              1.0535752e-01,  3.1186976e+00,  2.3081659e+01, -2.0144255e-01,
        #              3.3257127e-02,  2.9932617e-03,  1.4969879e+00,  3.5440731e+01,
        #             -1.5683125e-01, -4.0017970e-02, -4.4576611e-02,  2.8906257e+00,
        #              2.4773144e+01,  1.8246192e+00]

        
     
        # fab_std = [ 0.7077086,   0.70508003,  0.3104338,  16.11017   ,  6.2945657 ,  0.66618043,
        #             0.7445238,   0.34715655, 15.923401 ,   6.063388  ,  0.705116  ,  0.70655215,
        #             1.9695925,  10.840973  ,  3.1709101,   0.6837597 ,  0.72021097,  2.0887349,
        #            14.667681 ,   3.639706  ,  0.7222788,   0.69070977,  0.38787553, 16.480913,
        #             5.2354403,   0.7105775 ,  0.7003866,   0.99095803, 14.494578  ,  7.0202174 ]
        
        
        ################# Accel + Gyro
        # fab_mean = [ 1.7510343e-03,  7.0287548e-02,  1.1547769e+00,  1.5149701e+03,
        #              9.3159340e+01, -3.3905751e-03, -8.9651942e-02,  1.1833191e+00,
        #              1.0922792e+03,  8.1049934e+01,  8.1232406e-02,  1.2776729e-01,
        #              1.8254185e+00,  3.7115671e+02, -6.2096680e+01, -1.6829444e-02,
        #             -1.6615178e-01,  2.2893436e+00,  3.9469293e+02,  3.2716862e+01,
        #              7.5276539e-02, -1.1897401e-02,  1.1681240e+00,  1.5110760e+03,
        #             2.1992109e+01 ,-5.0007053e-02 ,-7.5436488e-02 , 1.2274623e+00,
        #             6.0448004e+02 , 1.4381465e+02]
        
        # fab_std = [6.2895179e-01, 7.7414805e-01, 7.8599490e-02, 4.6407816e+02, 4.6285602e+02,
        #            6.1407185e-01, 7.8379971e-01, 9.5540076e-02, 2.1040828e+02, 2.1277725e+02,
        #            6.9793510e-01, 6.9894773e-01, 6.6884267e-01, 1.4020927e+02, 2.3583398e+02,
        #            7.3753548e-01, 6.5332109e-01, 1.0165238e+00, 1.5833020e+02, 7.6413887e+01,
        #            7.9023337e-01, 6.0745394e-01, 7.8531422e-02, 5.1642279e+02, 8.1047565e+02,
        #            6.6943282e-01, 7.3651612e-01, 2.2264193e-01, 2.4494458e+02, 2.4368622e+02]

        
                  
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
