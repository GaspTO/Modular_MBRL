from node_module.best_first_node import BestFirstNode

class MCTSNode(BestFirstNode):
    def __init__(self):
        super().__init__()
        self._total_value = 0
        self._num_added_value = 0 #this is more robust than using the number of visits

    def get_total_value(self):
        return self._total_value

    def get_value(self):
        return (self._total_value)/(self._num_added_value)
    
    def add_value(self,delta_value):
        self._num_added_value += 1
        self._total_value += delta_value