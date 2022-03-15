from model_module.query_operations.reward_op import RewardOp
from model_module.query_operations.next_state_op import NextStateOp
from model_module.query_operations.state_value_op import StateValueOp
from model_module.query_operations.representation_op import RepresentationOp
from model_module.query_operations.mask_op import MaskOp
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

def mlp(
    input_size,
    layer_sizes,
    output_size,
    device,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]).to(device), act()] 
    return torch.nn.Sequential(*layers)


def cnn_dimensions(input_shape,kernel_sizes,num_channels=None):
    dims = [input_shape[1],input_shape[2]]
    for i in range(len(kernel_sizes)):
        if isinstance(kernel_sizes[i],int):
            kernel_sizes[i] = (kernel_sizes[i],kernel_sizes[i])
        dims[0] = dims[0] - (kernel_sizes[i][0]-1)
        dims[1] = dims[1] - (kernel_sizes[i][1]-1)
    return ([num_channels,*dims])



def cnn(
    input_shape:tuple,
    num_channels:list,
    kernel_sizes:List[Tuple],
    mlp_layer_sizes:list,
    device,
    activation=torch.nn.ReLU,
    output_activation=torch.nn.Identity,
):
    assert len(num_channels) == len(kernel_sizes)
    assert len(input_shape) == 3
    dims = cnn_dimensions(input_shape,kernel_sizes)
    assert dims[1] > 0 and dims[2] > 0
    

    stride = 1 #assuming stride 1 for now
    previous_channels = input_shape[0]
    layers = []
    number_of_layers = len(num_channels) + len(mlp_layer_sizes)
    for layer_num in range(number_of_layers):
        #Layers
        if layer_num < len(num_channels):
            layers += [nn.Conv2d(previous_channels, num_channels[layer_num], kernel_sizes[layer_num], stride=stride)]
            previous_channels = num_channels[layer_num]
        else:
            mlp_layer_num = layer_num-len(num_channels)
            layers += [torch.nn.Linear(mlp_layer_sizes[mlp_layer_num], mlp_layer_sizes[mlp_layer_num + 1]).to(device)]
        #Activations
        if layer_num < len(num_channels) + len(mlp_layer_sizes) - 1:
            if layer_num < len(num_channels):
                layers += [torch.nn.BatchNorm2d(num_channels[layer_num]),activation()]
            else:
                mlp_layer_num = layer_num-len(num_channels)
                layers += [torch.nn.BatchNorm1d(mlp_layer_sizes[mlp_layer_num]),activation()]
        else:
            layers += [output_activation()]
    return torch.nn.Sequential(*layers)

#example model = cnn((1,10,10),[100,300,500],[3,3,3],[],torch.device("cpu"))







class Disjoint_CNN(RepresentationOp,
                   StateValueOp,
                   MaskOp,
                   RewardOp,
                   NextStateOp,
                   torch.nn.Module):
    def __init__(
        self,
        observation_shape:tuple,
        action_space_size:int, 
        bool_normalize_encoded_states = False,
        optimizer = None,
        device = None
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print("Model is using "+str(self.device)+ " device")

        self.bool_normalize_encoded_states = bool_normalize_encoded_states
        if bool_normalize_encoded_states:
            raise ValueError("Normalization is still not available")
            
        self.action_space_size = action_space_size
        self.optimizer = optimizer

        self.observation_shape = observation_shape
        total_input_nodes = 1
        for x in observation_shape:
            total_input_nodes *= x


        ''' Initialize Networks'''
        encoding_channels = 1
        kernels = [3,3,3]
        self._representation_network = cnn(self.observation_shape,[100,300,encoding_channels],kernels,[],self.device)
        self.encoding_shape = cnn_dimensions(self.observation_shape,kernels,encoding_channels)
        self.encoding_nodes = np.array(self.encoding_shape).prod()
        ''' next hidden state'''
        self._dynamics_encoded_state_network = mlp(self.encoding_nodes + self.action_space_size,[],self.encoding_nodes,self.device)
        ''' reward '''
        self._dynamics_reward_network = mlp(self.encoding_nodes + self.action_space_size,[],1,self.device)
        ''' value '''
        self._prediction_value_network = mlp(self.encoding_nodes,[],1,self.device)
        ''' mask '''
        self._prediction_mask_network = mlp(self.encoding_nodes,[],action_space_size,self.device)

        print("done")

    ''' Representation '''
    def representation_query(self, observations:torch.Tensor, *keys):
        observations = observations.to(self.device).float()
        if isinstance(observations,np.ndarray):
            observations = torch.tensor(observations,device=self.device)
        assert observations.shape[1:] == self.observation_shape
        hidden_states = None
        ret_tuple = [] 
        for key in keys:
            v = self._representation_map(key,observations,hidden_states)
            ret_tuple.append(v)
        return tuple(ret_tuple)

    def _representation_map(self,key,observations:torch.Tensor,hidden_states): #! hidden_states=None
        if key == RepresentationOp.KEY:
            hidden_states = hidden_states if hidden_states is not None else self._representation_state(observations)
            return hidden_states
        else:
            raise ValueError("The key passed does not match any operation supported by the current model." + \
            "Choose one of the following: "+ str(RepresentationOp.KEY) + "\n")

    def _representation_state(self,observations:torch.Tensor):
        hidden_states = self._representation_network(observations)
        assert tuple(hidden_states.shape[1:]) == tuple(self.encoding_shape)
        hidden_states = hidden_states.view(hidden_states.shape[0],*self.encoding_shape)
        return hidden_states


    ''' Prediction '''
    def prediction_query(self, states: torch.Tensor, *keys):
        states = states.to(self.device)
        assert tuple(states.shape[1:]) == tuple(self.encoding_shape)
        ret_tuple = []
        for key in keys:
            v = self._prediction_map(key,states)
            ret_tuple.append(v)
        return tuple(ret_tuple)

    def _prediction_map(self,key,states:torch.Tensor):
        if key == StateValueOp.KEY:
            return self._prediction_value(states)
        elif key == MaskOp.KEY:
            return self._prediction_mask(states)
        else:
            raise ValueError("The key passed does not match any operation supported by the current model." + \
            "Choose one of the following: "+ str(StateValueOp.KEY))

    def _prediction_value(self,states:torch.Tensor):
        states = states.view(states.shape[0], -1)
        return self._prediction_value_network(states)

    def _prediction_mask(self,states:torch.Tensor):
        if self._prediction_mask_network is None:
            return torch.ones((states.shape[0],self.action_space_size),device=self.device)
        else:
            states = states.view(states.shape[0], -1)
            mask_logits = self._prediction_mask_network(states)
            return torch.sigmoid(mask_logits)

    ''' Dynamic '''
    def dynamic_query(self,states:torch.Tensor,actions:List[list],*keys):
        states = states.to(self.device)
        assert tuple(states.shape[1:]) == tuple(self.encoding_shape)
        ret_tuple = []
        for key in keys:
            v = self._dynamic_map(key,states,actions)
            ret_tuple.append(v)
        return tuple(ret_tuple)

    def _dynamic_map(self,key,states:torch.Tensor,actions:List[list]):
        if key == RewardOp.KEY:
            return self._dynamics_rewards(states,actions)
        elif key == NextStateOp.KEY:
            return self._dynamics_next_states(states,actions)
        else:
            raise ValueError("The key passed does not match any operation supported by the current model." + \
                "Choose one of the following: "+ str(RewardOp.KEY) + " or " + str(NextStateOp.KEY) + "\n")
        
    def _dynamics_next_states(self,encoded_states:torch.Tensor,actions:List[list]):
        encoded_states = encoded_states.view(encoded_states.shape[0], -1)
        input_vector = self._merge_dynamics_input(encoded_states,actions)
        next_states = self._dynamics_encoded_state_network(input_vector) 
        if self.bool_normalize_encoded_states:
            next_states = self._normalize_encoded_state(next_states)
        assert len(next_states.shape) == 2
        next_states = next_states.view(next_states.shape[0],*self.encoding_shape)
        return next_states 

    def _dynamics_rewards(self,encoded_states:torch.Tensor,actions:List[list]):
        encoded_states = encoded_states.view(encoded_states.shape[0], -1)
        input_vector = self._merge_dynamics_input(encoded_states,actions)
        if self._dynamics_reward_network is None:
            next_rewards = torch.zeros(input_vector.shape[0],device=self.device)
        else:
            next_rewards = self._dynamics_reward_network(input_vector)
        return next_rewards


    def _merge_dynamics_input(self, states:torch.tensor, actions:List[List]) -> torch.tensor: 
        '''
        This gets a tensor with all the states and the list of actions per state and calculates the transitions for them.
        e.g. states[0] is a state and actions[0] is a list of all the actions the transitions will be calculates
        '''
        assert states.shape[0] == len(actions), "There needs to be a list of actions per encoded state"
        assert len(states.shape) >= 2, "enconded_states needs to have a batch dimension"
        input_vector = []
        for state_idx in range(states.shape[0]):
            for action in actions[state_idx]:
                action_one_hot = torch.zeros(self.action_space_size,device=self.device).float()
                action_one_hot[action] = 1
                x = torch.cat((states[state_idx],action_one_hot))
                input_vector.append(x.unsqueeze_(0))
        input_vector = torch.cat(input_vector)
        return input_vector


    ''' instance specific '''
    def get_optimizers(self)->list:
        if self.optimizer is None:
            self.optimizer =  torch.optim.Adam(self.parameters(),weight_decay=1e-04)
        return [self.optimizer]

    def get_schedulers(self)->list:
        self.schedulers = []
        return self.schedulers


