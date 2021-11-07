
from environments.cart_pole import CartPole
from environments.minigrid import Minigrid
from environments.tictactoe import TicTacToe

from planning_module.abstract_breadth_first_search import AbstractBreadthFirstSearch
from planning_module.minimax import Minimax
from planning_module.average_minimax import AverageMinimax
from planning_module.ucb_best_first_minimax import UCBBestFirstMinimax
from planning_module.ucb_monte_carlo_tree_search import UCBMonteCarloTreeSearch

from policy_module.epsilon_greedy_value import EpsilonGreedyValue
from policy_module.epsilon_greedy_visits import EpsilonGreedyVisits
from policy_module.visit_ratio import VisitRatio

from utils.optimization.simple_optimizer import SimpleOptimizer
from utils.storage.proportional_priority_buffer import ProportionalPriorityBuffer
from utils.storage.uniform_buffer import UniformBuffer

from loss_module.monte_carlo_mvr import MonteCarloMVR
from loss_module.offline_td_mvr import OfflineTDMVR
from loss_module.online_td_mvr import OnlineTDMVR

from model_module.disjoint_mlp import Disjoint_MLP

from torch.utils.tensorboard import SummaryWriter, writer
from datetime import datetime


from math import sqrt
import os, psutil
import gc
from pympler import asizeof, tracker
import datetime


        


now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

''' environment '''
environment = input("choose environment:\n 1 - cartpole\n 2 - minigrid\n 3 - tictactoe\n")
if environment == "1":
    print("Cartpole chosen")
    environment = CartPole(500)
elif environment == "2":
    print("Minigrid chosen")
    environment = Minigrid(max_steps=18)
elif environment == "3":
    print("Tictactoe chosen")
    environment = TicTacToe(self_play=False)
else:
    print("couldn't understand choice. Choosing cartpole by default.")
    environment = CartPole(500)


experiment_name = str(environment) + "_" + time_str
writer = SummaryWriter(log_dir="logs/runs/"+str(time_str)+ "_" + str(experiment_name))



model = Disjoint_MLP(
    observation_shape = environment.get_input_shape(),
    action_space_size = environment.get_action_size(),
    encoding_shape = (8,),
    fc_reward_layers = [300],
    fc_value_layers =  [300], 
    fc_representation_layers = [300],
    fc_dynamics_layers = [300],
    fc_mask_layers = [300],
    bool_normalize_encoded_states = False 
)


action_size = environment.get_action_size()
num_of_players = environment.get_num_of_players()

''' planning '''
planning = input("choose planning:\n 1 - minimax\n 2 - averaged minimax\n 3 - BFMMS\n 4 - MCTS\n")
if planning == "1":
    print("Minimax chosen")
    planning = Minimax(model,action_size,num_of_players,max_depth=3,invalid_penalty=-1)
elif planning == "2":
    print("Averaged Minimax chosen")
    planning = AverageMinimax(model,action_size,num_of_players,max_depth=3,invalid_penalty=-1)
elif planning == "3":
    print("BFMMS chosen")
    planning = UCBBestFirstMinimax(model,action_size,num_of_players,num_iterations=15,search_expl=sqrt(2),invalid_penalty=-1)
elif planning == "4":
    print("MCTS chosen")
    planning = UCBMonteCarloTreeSearch(model,action_size,num_of_players,num_iterations=15,search_expl=sqrt(2),invalid_penalty=-1)
else:
    print("couldn't understand choice. Choosing Minimax by default.")
    planning = Minimax(model,action_size,num_of_players,max_depth=3,invalid_penalty=-1)

''' policy '''
policy = input("choose policy:\n 1 - epsilon value greedy\n 2 - epsilon visit greedy\n 3 - visit distribution\n")
if policy == "1":
    print("Epsilon value greedy chosen")
    policy = EpsilonGreedyValue(environment,planning,epsilon=0.05,reduction='root')
elif policy == "2":
    if isinstance(planning,AbstractBreadthFirstSearch):
        raise ValueError("A breadth first search algorithm can not be paired with a strategy that requires node visits")
    print("Epsilon visit greedy chosen")
    policy = EpsilonGreedyVisits(environment,planning,epsilon=0.05,reduction='root')
elif policy == "3":
    if isinstance(planning,AbstractBreadthFirstSearch):
        raise ValueError("A breadth first search algorithm can not be paired with a strategy that requires node visits")
    print("Visit ration chosen")
    policy = VisitRatio(environment,planning,temperature=0.05,reduction='root')
else:
    print("couldn't understand choice. Choosing Epsilon value greedy by default.")
    policy = EpsilonGreedyValue(environment,planning,epsilon=0.05,reduction='root')

''' loss '''
loss_module = input("choose loss:\n 1 - Monte Carlo\n 2 - Offline TD\n 3 - Online TD\n")
if loss_module == "1":
    print("Monte Carlo chosen")
    loss_module = MonteCarloMVR(model,unroll_steps=5,gamma_discount=0.99)
elif loss_module == "2":
    print("Offline TD chosen")
    loss_module = OfflineTDMVR(model,unroll_steps=5,n_steps=1,gamma_discount=0.99)
elif loss_module == "3":
    print("Online TD chosen")
    loss_module = OnlineTDMVR(model,unroll_steps=5,n_steps=1,gamma_discount=0.99) 
else:
    print("couldn't understand choice. Choosing Monte Carlo by default.")
    loss_module = MonteCarloMVR(model,unroll_steps=5,gamma_discount=0.99)
    
''' optimizer '''
optimizer = SimpleOptimizer(model.parameters(),model.get_optimizers(),model.get_schedulers(),max_grad_norm=20)

''' storage '''
storage = input("choose storage:\n 1 - uniform buffer \n 2 - priority buffer\n")
if storage == "1":
    storage = UniformBuffer(max_buffer_size=1000)
elif storage == "2":
    storage = ProportionalPriorityBuffer(max_buffer_size=1000)
else:
    print("Couldn't understand choice. Choosing uniform by default.")
    storage = UniformBuffer(max_buffer_size=1000)


episodes = int(input("how many episodes?\n"))
updates_per_episode = int(input("how many updates per episode?\n"))
batch_size = int(input("What's the update batch size\n"))
scores = []
for ep in range(episodes):
    game = policy.play_game()

            
    #! log score
    score = sum(game.rewards)
    print("episode:"+str(ep)+ " score:"+str(score))
    scores.append(score)
    scores = scores[-100:]
    writer.add_scalar("Score/avg_100_score",sum(scores)/len(scores),ep)
    writer.add_scalar("Score/score",score,ep)


    #! store game
    if isinstance(storage,ProportionalPriorityBuffer):
        new_loss, new_info = loss_module.get_loss(game.nodes) 
        storage.add(game.nodes,new_info["loss_per_node"])
    else:
        storage.add(game.nodes)
        
    #! learn
    for lep in range(updates_per_episode):
        nodes = storage.sample(batch_size)
        loss, info = loss_module.get_loss(nodes)
        optimizer.optimize(loss)    
        if isinstance(storage,ProportionalPriorityBuffer): 
            storage.updated_priorities(nodes,info["loss_per_node"])
        writer.add_scalar("Loss/loss",loss,ep)
        writer.add_scalar("Loss/loss_value",info["loss_value"],ep)
        writer.add_scalar("Loss/loss_reward",info["loss_reward"],ep)
        writer.add_scalar("Loss/loss_mask",info["loss_mask"],ep)
        
writer.close()



