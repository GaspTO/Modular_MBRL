from planning_module.planning import Planning
from math import sqrt, log

class AbstractBestFirstSearch(Planning):
    def __init__(self,model):
        super().__init__(model)

    '''
        Node value methods
    ''' 
    def _get_ucb_best_node(self,node,exploration):
        """ returns (1) the best node based on the ucb with mask penalty
                    (2) its action
                    (3) the unmasked ucb value
                    (4) the value with mask penalty """
        best_masked_value,best_child = float("-inf"),None,
        for action,child in node.get_children().items():
            masked_penalized_child_value, _ = self._ucb_function(child,exploration)
            if masked_penalized_child_value > best_masked_value:
                best_masked_value,best_child = masked_penalized_child_value,child
        return best_child
        
    def _ucb_function(self,child,c):
        parent = child.get_parent()
        child_action = child.get_parent_action()
        value_for_parent = parent.successor_value(child_action)
        if c != 0:
            exploration = sqrt(log(parent.get_visits())/(child.get_visits()+ 1))
        else:
            exploration = 0
        unmasked_value = value_for_parent + c * exploration
        valid_value = parent.get_action_mask()[child_action].item()
        return self._value_after_mask(unmasked_value,valid_value), unmasked_value

    def _value_after_mask(self,value,valid):
        return (value * valid) + (1-valid) * self.invalid_penalty


