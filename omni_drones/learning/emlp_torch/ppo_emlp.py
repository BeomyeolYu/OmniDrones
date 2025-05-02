import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .groups import *
from .reps import *
from .nn import EMLPBlock, Linear, uniform_rep
from .groups import SO2eR3, Trivial
from .reps import Vector, Scalar
from .spectral_norm_regularization import spectral_norm

###################################################################################
############## Monolithic Architecture ############################################
###################################################################################
class EMLP_MONO_Actor_PPO(nn.Module):
    def __init__(self, action_dim, actor_hidden_dim, device, hidden_num=2, log_std=0):
        """
        Equivariant MLP-based monolithic actor network for PPO.

        Args:
            action_dim: 
            actor_hidden_dim: 
            device:
            hidden_num : Number of hidden layers.
            log_std: Initial log standard deviation for the action distribution.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
                                                  <--------- ρ_θ(g)R ---------> 
            Output: ρ_e(g)f, ρ_e(g)M
        """
        super().__init__()
        self.device = device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_trivialR3 = Trivial(3).to(self.device)  # ρ_e(g) for 3D vector values
        
        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
        self.rep_in  = Vector(self.G_SO2eR3)*6 + Scalar(self.G_trivialR1)*2 + Vector(self.G_trivialR3)
        # Output representation: ρ_e(g)f, ρ_e(g)M
        self.rep_out = Scalar(self.G_trivialR1) + Vector(self.G_trivialR3)
        
        # Define the hidden layers based on the number of hidden layers and size
        middle_layers = hidden_num*[uniform_rep(actor_hidden_dim, self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)
        # Output layers for log standard deviation of the action distribution
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

        # Apply weight initialization
        self.network[-1].weight.data.mul_(0.1)
        self.network[-1].bias.data.mul_(0.0)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input observation (ex, eIx, ev, R, eb1, eIb1, eΩ).

        Returns:
            action: The action (f, M) in body-fixed frame.
        """
        mean = torch.tanh(self.network(state))
        log_std = self.log_std.expand_as(mean)  # Expand log_std to match mean's shape
        std = torch.exp(log_std)  # Convert log standard deviation to standard deviation
        return mean, std

    # def get_dist(self, x):
    #     """
    #     Compute the Gaussian action distribution given the input state.
        
    #     Args:
    #         x: Input state tensor.

    #     Returns:
    #         normal: A Normal distribution representing the policy.
    #     """
    #     mean, std = self.forward(x)
    #     return Normal(mean, std)

    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)
    

class EMLP_MONO_Critic_PPO(nn.Module):
    def __init__(self, critic_hidden_dim, device, hidden_num=2):
        """
        Equivariant MLP-based monolithic critic network for PPO.

        Args:
            critic_hidden_dim: 
            device:
            hidden_num : Number of hidden layers.

        Group representation:
            Input: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
                                                  <--------- ρ_θ(g)R ---------> 
            Output: ρ_e(g)V
        """
        super().__init__()
        self.device = device

        # Define groups
        self.G_SO2eR3 = SO2eR3().to(self.device)  # ρ_θ(g) for 3D rotations
        self.G_trivialR1 = Trivial(1).to(self.device)  # ρ_e(g) for scalar values
        self.G_trivialR3 = Trivial(3).to(self.device)  # ρ_e(g) for 3D vector values

        # Input representation: ρ_θ(g)ex, ρ_θ(g)eIx, ρ_θ(g)ev, ρ_θ(g)b1, ρ_θ(g)b2, ρ_θ(g)b3, ρ_e(g)eb1, ρ_e(g)eIb1, ρ_e(g)eΩ
        self.rep_in  = Vector(self.G_SO2eR3)*6 + Scalar(self.G_trivialR1)*2 + Vector(self.G_trivialR3)
        # Output representation: ρ_e(g)V(s)
        self.rep_out = Scalar(self.G_trivialR1) 
        
        # Define the hidden layers for the critic network
        middle_layers = hidden_num*[uniform_rep(critic_hidden_dim, self.G_SO2eR3)]
        reps = [self.rep_in]+middle_layers

        # Build the network as a sequence of EMLP blocks
        self.network = torch.nn.Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],self.rep_out)
        ).to(self.device)


    def forward(self, state):
        """
        Forward pass through the critic network to estimate the state value.
        
        Args:
            state: Input state tensor.

        Returns:
            Estimated value function V(s).
        """
        return self.network(state)
    
    def spectral_norm_regularization(self):
        """
        Apply spectral normalization regularization to the network weights.
        """
        return spectral_norm(self.network, self.device)