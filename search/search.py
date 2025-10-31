# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    from util import Stack

    # Cria uma pilha para armazenar nós a explorar.
    # Cada item empilhado é uma tupla: (estado_atual, caminho_de_acoes_para_chegar_ate_este_estado)
    stack = Stack()

    # Conjunto para registrar estados já visitados (graph-search)
    visited = set()

    # Empilha o estado inicial com um caminho vazio (nenhuma ação realizada ainda)
    stack.push((problem.getStartState(), []))

    # Loop até que não haja mais nós para expandir
    while not stack.isEmpty():
        # Desempilha o próximo nó a ser explorado
        state, actions = stack.pop()

        # Se o estado desempilhado for objetivo, retornamos o caminho de ações que levou até ele
        if problem.isGoalState(state):
            return actions

        # Se ainda não visitamos este estado, expandimos seus sucessores
        if state not in visited:
            # Marca o estado como visitado para evitar ciclos
            visited.add(state)
            # Itera sobre os sucessores do estado atual
            for successor, action, _ in problem.getSuccessors(state):
                # Para DFS, empilhamos cada sucessor com o caminho atualizado (concatena a ação)
                stack.push((successor, actions + [action]))

    # Se não encontrado, retorna lista vazia (sem solução)
    return []
    ###util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue
    start_state = problem.getStartState()
    # Cria uma fila para armazenar nós a explorar.
    # Cada item enfileirado é uma tupla: (estado_atual, caminho_de_acoes_para_chegar_ate_este_estado)
    queue = Queue()

    # Conjunto para registrar estados já visitados (graph-search)
    visited = set()

    # Enfileira o estado inicial com um caminho vazio
    queue.push((start_state, []))

    # Loop até que a fila esteja vazia
    while not queue.isEmpty():
        # Desenfileira o próximo nó a ser explorado
        state, actions = queue.pop()

        # Se o estado desenfileirado for objetivo, retornamos o caminho de ações
        if problem.isGoalState(state):
            return actions

        # Se ainda não visitamos este estado, expandimos seus sucessores
        if state not in visited:
            # Marca o estado como visitado
            visited.add(state)
            # Para cada sucessor do estado atual
            for successor, action, _ in problem.getSuccessors(state):
                # Enfileira o sucessor com o caminho atualizado (concatena a ação)
                queue.push((successor, actions + [action]))
    return []
    ###util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    start_state = problem.getStartState()
    # Cria uma fila de prioridade onde cada entrada é ((estado, caminho, custo), prioridade)
    pq = PriorityQueue()

    # Conjunto para registrar estados visitados (graph-search)
    visited = set()

    # Insere o estado inicial com custo 0; prioridade igual ao custo atual
    pq.push((start_state, [], 0), 0)

    # Loop enquanto houver nós na fila de prioridade
    while not pq.isEmpty():
        # Remove o item com menor prioridade (menor custo acumulado)
        state, actions, cost = pq.pop()

        # Se o estado é objetivo, retornamos o caminho para ele
        if problem.isGoalState(state):
            return actions

        # Se ainda não visitamos este estado, expandimos
        if state not in visited:
            # Marca como visitado
            visited.add(state)
            # Para cada sucessor do estado atual
            for successor, action, stepCost in problem.getSuccessors(state):
                # Calcula o novo custo acumulado até o sucessor
                newCost = cost + stepCost
                # Insere/atualiza o sucessor na fila de prioridade com prioridade = novo custo
                pq.push((successor, actions + [action], newCost), newCost)

    return []
    ###util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # Se nenhuma heurística for passada, usa heurística nula (0)
    from util import PriorityQueue
    if heuristic is None:
        heuristic = lambda state, problem=None: 0

    # Cria a fila de prioridade para nós a explorar
    # Cada elemento: (estado, caminho, custo_g)
    pq = PriorityQueue()

    # Conjunto de estados visitados (graph-search)
    visited = set()

    # Calcula prioridade inicial: custo_g (0) + heurística do estado inicial
    start = problem.getStartState()
    startPriority = heuristic(start, problem)
    pq.push((start, [], 0), startPriority)

    # Loop até esgotar a fila
    while not pq.isEmpty():
        # Pop do menor f = g + h
        state, actions, cost = pq.pop()

        # Se alcançamos o objetivo, retornamos o caminho
        if problem.isGoalState(state):
            return actions

        # Se ainda não visitado, expandimos
        if state not in visited:
            visited.add(state)
            # Para cada sucessor, calcula novo g e prioridade f = g + h
            for successor, action, stepCost in problem.getSuccessors(state):
                newCost = cost + stepCost
                priority = newCost + heuristic(successor, problem)
                pq.push((successor, actions + [action], newCost), priority)

    return []
    ###util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
