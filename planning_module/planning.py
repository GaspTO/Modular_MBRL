from node_module.node import Node

class Planning:
    def __init__(self,model):
        self.model = model
        self.info = {}

    def plan(self,observation,player,mask) -> Node:
        '''
            creates a node with the observation and runs the search algorithm.
            returns the node
        '''
        raise NotImplementedError
        
        



