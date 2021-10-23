from planning_module.minimax import Minimax

class AverageMinimax(Minimax):
    def __init__(self,
    model,
    action_size,
    num_of_players,
    max_depth,
    invalid_penalty):
        super().__init__(
            model=model,
            action_size=action_size,
            num_of_players=num_of_players,
            max_depth=max_depth,
            invalid_penalty=invalid_penalty)
        
    def _update_node(self,node):
        value = self._get_children_average_successor_value(node)
        node.set_value(value)
        
    def _get_children_average_successor_value(self,node):
        total_value = 0.
        total_mask = 0.
        mask = node.get_action_mask()
        for child in node.get_children_nodes():
            total_value += node.successor_value(child.get_parent_action()) * mask[child.get_parent_action()].item() 
            total_mask += mask[child.get_parent_action()].item()
        return total_value/total_mask if total_mask != 0. else 0.

        

