from planning_module.planning import Planning

class AbstractDepthFirstSearch(Planning):
    def __init__(self,model):
        super().__init__(model)