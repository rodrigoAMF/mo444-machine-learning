import numpy as np
import random
from collections import namedtuple, deque

from game import Agent
from pacman import Directions

#import torch
#import torch.nn.functional as F
#import torch.optim as optim

#UPDATE_EVERY = 4  # how often to update the network

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

action_to_direction = {
    0: Directions.NORTH,
    1: Directions.SOUTH,
    2: Directions.EAST,
    3: Directions.WEST,
    4: Directions.STOP
}

direction_to_action = {
    Directions.NORTH: 0,
    Directions.SOUTH: 1,
    Directions.EAST: 2,
    Directions.WEST: 3,
    Directions.STOP: 4
}

debug = False

class gaAgent(Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, params, layout_used, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.params = params
        self.seed = random.seed(seed)
        self.features = np.ones(11)

        #if debug:
        #    print("population")
        #    print(self.population.shape)
        #    print(self.population)

        ## Q-Network
        #if layout_used == "smallClassic":
        #    self.qnetwork_local = QNetworkSmall(state_size, action_size, seed).to(device)
        #    self.qnetwork_target = QNetworkSmall(state_size, action_size, seed).to(device)
        #elif layout_used == "mediumClassic":
        #    self.qnetwork_local = QNetworkMedium(state_size, action_size, seed).to(device)
        #    self.qnetwork_target = QNetworkMedium(state_size, action_size, seed).to(device)
        #elif layout_used == "originalClassic":
        #    self.qnetwork_local = QNetworkOriginal(state_size, action_size, seed).to(device)
        #    self.qnetwork_target = QNetworkOriginal(state_size, action_size, seed).to(device)
        #else:
        #    raise ValueError("Unkown layout", layout_used)
        #self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=params["lr"],
        #                            weight_decay=params["weight_decay"], amsgrad=True)

        # Replay memory
        #self.memory = ReplayBuffer(state_size, action_size, self.params["buffer_size"], self.params["batch_size"], seed)
        # Initialize time step (for updating every params["update_every"] steps)
        #self.t_step = 0

    #def step(self, state, action, reward, next_state, done):
    #    self.state = state
        
        #print("state:")
        #print(state.shape)
        #print(state)
        #assert False

        # Save experience in replay memory
        #self.memory.add(state, action, reward, next_state, done)

        # Learn every params["update_every"] time steps.
        #self.t_step = (self.t_step + 1) % self.params["update_every"]
        #if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
        #    if len(self.memory) > self.params["batch_size"]:
        #        experiences = self.memory.sample()
        #        self.learn(experiences, self.params["gamma"])
    #    return

    def getAction(self, state, legal_actions, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """

        # feature size 11
        # possible actions 5

        #print(f"individual {self.individual}")
        #print(f"features {self.features}")
        
        #print(f"legal_actions {legal_actions}")
        legal_actions = np.array([direction_to_action[action] for action in legal_actions]) 
        #print(f"legal_actions {legal_actions}")

        weighted_actions = np.zeros(5)
        for i in range(5):
            weighted_features = np.dot(self.features, self.individual[11*i:11*(i+1)])
            weighted_actions[i] = np.sum(weighted_features)
        #print(f"weighted_actions {weighted_actions}")
        possible_actions = weighted_actions[legal_actions]
        #print(f"possible_actions {possible_actions}")
        best_action_index = np.argmax(possible_actions)
        #print(f"best_action_index {best_action_index}")
        action = legal_actions[best_action_index]
        #print(f"action {action}")
        action = action_to_direction[action]
        #print(f"action {action}")

        #assert False
        #action = random.choice(legal_actions)
        return action

    '''
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params["tau"])
    '''

    #def soft_update(self, local_model, target_model, tau):
    #    """Soft update model parameters.
    #    θ_target = τ*θ_local + (1 - τ)*θ_target
    #    Params
    #    ======
    #        local_model (PyTorch model): weights will be copied from
    #        target_model (PyTorch model): weights will be copied to
    #        tau (float): interpolation parameter
    #    """
    #    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)




# ------------------------------------------------------------------------------------------

"""
def generateGene():
    '''Função que retorna um gene. Ela gera uma lista com 5 números de 0 a 100 criados aleatoriamente'''
    gene=[]
    for weight in range(0,5):
        gene.append(random.uniform(0, 100))
    return gene

def generateIndividuo():
    '''Função que retorna um indivíduo. Ela gera um indivíduo com 1000 passos, valor suficiente para resolver
    até o layout mais complexo do pacman.'''
    individuo=[]
    for jogadas in range(0,1000):
        individuo.append(generateGene())
    return individuo

def generatePopulation(size):
    '''Função que recebe o layout do jogo e um tamanho e cria uma população de indivíduos. Ela
    retorna duas listas, a população e os scores de cada indivíduo. O score na posição 0 da lista será do 
    indivíduo da população na posição 0.'''
    population=[]
    for i in range(0,size):
        individuo = generateIndividuo()
        individuo_temp=individuo #Passamos para uma variável temporária pois o .pop(0)do getAction
                                    #remove os passos dados dos indivíduos
        population.append(individuo)
    return population

def pacmanAvarageScore(individuo,num_tentativas,flag):
    '''Função que recebe o indivíduo, a quantidade de jogos que ele irá realizar e o layout do jogo.
    Ela retorna o score médio da quantidade de jogos realizados para esse indivíduo.'''
    tentativas_ind=[]
    layouts=['originalClassic']
    for layout in layouts:
        for tentativas in range(0,num_tentativas):
            if flag:
                args = ['--layout',layout,'--pacman','DumbAgent','-q']
            else:
                args = ['--layout',layout,'--pacman','DumbAgent']
            args_list = readCommand(args)
            test = GAAgent(individuo)
            score= pacman.runGames(pacman=test,layout=args_list['layout'],ghosts=args_list['ghosts'],display=args_list['display'],
                           numGames=args_list['numGames'],record=args_list['record'])
            tentativas_ind.append(score)
    return np.mean(tentativas_ind)

def evaluatePopulation(population,flag):
    '''Função que recebe o layout do jogo e um tamanho e cria uma população de indivíduos. Ela
    retorna duas listas, a população e os scores de cada indivíduo. O score na posição 0 da lista será do 
    indivíduo da população na posição 0.'''
    scores=[]
    for i in range(0,len(population)):
        print('individuo ',i+1)
        individuo = population[i]
        individuo_temp=deepcopy(individuo) #Passamos para uma variável temporária pois o .pop(0)do getAction
                                    #remove os passos dados dos indivíduos
        avg_score = pacmanAvarageScore(individuo_temp,5,flag)
        scores.append(avg_score)
    return scores

def mutation(population,rate):
    '''Função que recebe uma população e uma taxa e define uma porcentagem da população para realizar mutação
    de cromossomos nos seus indivíduos da essa taxa recebida'''
    for selected in range(int(len(population)*rate)): #Seleciona uma quantidade x de indivíduos, dado a rate
        position_individual = random.randint(0,len(population)-1) #seleciona um indivíduo aleatório na população
        position_cromossom1 = random.randint(0,len(population[position_individual])-1)#Seleciona um cromossomo aleatório
        position_cromossom2 = random.randint(0,len(population[position_individual])-1)#Seleciona outro cromossomo aleatório
        crom1 = population[position_individual][position_cromossom1]
        crom2 = population[position_individual][position_cromossom2]
        
        population[position_individual][position_cromossom1] = crom2
        population[position_individual][position_cromossom2] = crom1
    return population

def torneio(pairs,tournment_size):
    participantes_torneio=[]
    for number in range(tournment_size):
        position_individual = random.randint(0,len(pairs)-1)
        individuo = pairs[position_individual]
        participantes_torneio.append(individuo)
    participantes_torneio.sort(reverse=True, key=lambda li: li[0])
    return participantes_torneio[0]

def crossover(pairs,range_cromossomos,aleatorio):
    '''Função que recebe um pair [score, população] e uma taxa e um range de valores e define uma porcentagem da população
    para realizar crossover desse range de valores (recebido pela função) para os cromossomos dos indivíduos'''
    novos_pares=[]
    for i in range(int((len(pairs)*0.8)/2)): #Para 80% dosindivíduos de individuos na populacao
        
        individual1 = torneio(pairs,4) #seleciona um indivíduo 1 aleatório na população
        individual2 = torneio(pairs,4) #seleciona um indivíduo 2 aleatório na população
        #Apos selecionar as posições dos indivíduos, selecionamos o range de seus primeiros cromossomos     
        crom1_list = individual1[1][aleatorio:aleatorio+range_cromossomos]
        crom2_list = individual2[1][aleatorio:aleatorio+range_cromossomos]
            
        individual1[1][aleatorio:aleatorio+range_cromossomos] = crom2_list
        individual2[1][aleatorio:aleatorio+range_cromossomos] = crom1_list
        novos_pares.append(individual1)
        novos_pares.append(individual2)
    population = [el[1] for el in novos_pares]
    return population

def makePairs(scores,population):
    lista=[]
    for i in range(len(population)):
        item=[scores[i], population[i]]
        lista.append(item)
    lista.sort(reverse=True, key=lambda li: li[0])
    return lista

def mergePopulationAndScores(pairs,pairs_linha):
    pop = pairs[:5] + pairs_linha[:-5]
    population = [el[1] for el in pop]
    scores = [el[0] for el in pop]
    return population, scores
"""