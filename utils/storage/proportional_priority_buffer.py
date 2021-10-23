import random
import numpy as np
import torch

'''
*   This is the proportional implementation of the priority replay buffer - https://arxiv.org/pdf/1511.05952.pdf
*
*   Every time a game is added a priority associated with it has to be passed.
*   The higher the priority, the higher the probability it will be sampled from the buffer
*   In the original article, this was the temporal difference error.
*
*   As training goes by, it's important to update the priority of the games. Call update_priority for that
*      
*   This implementation allows to control 2 properties:
*       1) priority decay allows for updates to not completely replace the previous priority, but slowly change it using 
*           the formula:  (decay) * new_priority + (1-decay) * previous_priority   -- This is not described in the paper
*       
*       2) alpha allows to regulate the importance of the priority of a game when calculating its probability
*        of being sampled. The conversion of priority to probability is:
*
*          probability =  (priority ** alpha) / Σ (priority_k ** alpha)
*
'''
class ProportionalPriorityBuffer():
    PriorityKey = "ProportionalBufferPriority"
    def __init__(self,max_buffer_size,priority_decay=1,alpha=1,beta=1,minimum_priority=0.001):
        self.buffer = []
        self.trainer = None
        self.max_buffer_size = max_buffer_size
        self.priority_decay = priority_decay
        self.alpha = alpha
        self.beta = beta
        self.minimum_priority = minimum_priority
        self.num = 0

    def add(self,nodes:list,priorities:list=None):
        if priorities is None: priorities = [0] * len(nodes)
        assert len(nodes) == len(priorities), "the number of nodes and number of priorities inserted needs to be the same"
        for i in range(len(nodes)):
            self._add_node(nodes[i],priorities[i])           

    def _add_node(self,node,priority=0):
        self._update_priority(node,priority) 
        self.buffer.append(node)
        self.num += 1
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        assert len(self.buffer) <= self.max_buffer_size

    def sample(self, num=1):
        node_priorities = []
        for idx, node in enumerate(self.buffer):
            priority = node.info[self.PriorityKey]**self.alpha + self.minimum_priority
            node_priorities.append(priority)
        node_priorities = np.array(node_priorities,dtype="float32")
        node_probabilities = node_priorities / np.sum(node_priorities)
        number_of_nodes = min(num,len(self.buffer))
        nodes = np.random.choice(self.buffer, number_of_nodes, p=node_probabilities,replace=False)
        return list(nodes)

    def updated_priorities(self,nodes:list,priorities:list):
        for i in range(len(nodes)):
            self._update_priority(nodes[i],priorities[i])

    def _update_priority(self,node,priority):
        ''' priority is usually the loss or td error...'''
        if isinstance(priority,torch.Tensor): priority = priority.item()
        if self.PriorityKey not in node.info:
            node.info[self.PriorityKey] = priority
        else:
            node.info[self.PriorityKey] = self.priority_decay * priority + (1-self.priority_decay) * node.info[self.PriorityKey]

    def get_size(self):
        return len(self.buffer)

    def get_num_of_added_samples(self):
        return self.num

    def __str__(self):
        return "ProportionalPrioritizedBuffer"+str(self.max_buffer_size)+"_alpha"+str(self.alpha)


