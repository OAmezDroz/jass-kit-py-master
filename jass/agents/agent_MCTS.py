import numpy as np
import random
from jass.game.const import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.agents.agent import Agent
from jass.game.rule_schieber import RuleSchieber

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def uct_value(self, exploration_param=1.4):
        if self.visits == 0:
            return float('inf')  # unerforschte Knoten bevorzugen
        return (self.wins / self.visits) + exploration_param * np.sqrt(np.log(self.parent.visits) / self.visits)

class AgentMCTS(Agent):
    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()

    def run_simulation(self, node):
        """
        Führe eine Simulation von dem aktuellen Knoten aus, um ein Spielende zu erreichen.
        """
        sim_game = self.game_simulator
        sim_game.init_from_game_state(node.state)

        # Zufaellige Spielausgänge simulieren
        while not sim_game.is_done():
            valid_cards = self._rule.get_valid_cards_from_obs(sim_game.current_player)
            action = random.choice(valid_cards)
            sim_game.play_card(action)

        # Ergebniss des Spiels
        result = sim_game.get_winner()
        if result == 0:  # wenn Agent gewinnt
            return 1
        return 0

    def select_best_action(self, node):
        """
        Wähle die Aktion mit dem höchsten UCT-Wert.
        """
        best_child = max(node.children, key=lambda child: child.uct_value())
        return best_child.action

    def action_trump(self, obs: GameObservation) -> int:
        """
        Wähle Trumpf mit MCTS basierend auf der Hand und der Situation.
        """
        root_node = MCTSNode(state=obs)

        for _ in range(self.iterations):
            node = root_node
            # Auswahl
            while node.children:
                node = max(node.children, key=lambda child: child.uct_value())
            
            # Expansion
            if node.visits > 0:
                for action in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE, PUSH]:
                    child_state = obs.clone()
                    child_state.trump = action
                    child_node = MCTSNode(state=child_state, parent=node, action=action)
                    node.add_child(child_node)
                node = random.choice(node.children)

            # Simulation
            result = self.run_simulation(node)
            
            # Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent

        return self.select_best_action(root_node)

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Wähle die beste Karte basierend auf MCTS.
        """
        root_node = MCTSNode(state=obs)

        for _ in range(100):
            node = root_node

            # Auswahl
            while node.children:
                node = max(node.children, key=lambda child: child.uct_value())

            # Expansion
            if node.visits > 0:
                valid_cards = self._rule.get_valid_cards_from_obs(obs)
                for action in valid_cards:
                    child_state = obs.clone()
                    child_state.play_card(action)
                    child_node = MCTSNode(state=child_state, parent=node, action=action)
                    node.add_child(child_node)
                node = random.choice(node.children)

            # Simulation
            result = self.run_simulation(node)

            # Backpropagation
            while node is not None:
                node.update(result)
                node = node.parent

        return self.select_best_action(root_node)
