import logging
import numpy as np
from jass.agents.agent_cheating import AgentCheating
from jass.game.const import card_strings, PUSH, MAX_TRUMP
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber

class AgentMinimax(AgentCheating):
    def __init__(self, depth=3):
        super().__init__()
        self.depth = depth  # Tiefe für den Minimax-Baum
        self._logger = logging.getLogger(__name__)
        self._rule = RuleSchieber()

    def action_trump(self, state: GameState) -> int:
        """
        Wählt Trumpf basierend auf einer festen Strategie oder zufällig.
        """
        # Beispielhafte Entscheidung: Trumpf zufällig wählen
        if state.forehand == -1:
            return PUSH if np.random.choice([True, False]) else int(np.random.randint(0, MAX_TRUMP))
        return int(np.random.randint(0, MAX_TRUMP))

    def action_play_card(self, state: GameState) -> int:
        """
        Wählt die beste Karte basierend auf Minimax mit Alpha-Beta-Pruning.
        """
        valid_cards = self._rule.get_valid_cards_from_state(state)
        best_score = float('-inf')
        best_card = None

        for card in np.flatnonzero(valid_cards):
            new_state = state.clone()
            new_state.play_card(card)
            score = self.minimax(new_state, self.depth, False, float('-inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_card = card

        self._logger.info(f"Played card: {card_strings[best_card]}")
        return best_card

    def minimax(self, state: GameState, depth, maximizing_player, alpha, beta) -> int:
        """
        Minimax mit Alpha-Beta-Pruning.
        """
        if depth == 0 or state.is_done():
            return self.evaluate_state(state)

        valid_cards = self._rule.get_valid_cards_from_state(state)
        if maximizing_player:
            max_eval = float('-inf')
            for card in np.flatnonzero(valid_cards):
                new_state = state.clone()
                new_state.play_card(card)
                eval_score = self.minimax(new_state, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta-Abschneidung
            return max_eval
        else:
            min_eval = float('inf')
            for card in np.flatnonzero(valid_cards):
                new_state = state.clone()
                new_state.play_card(card)
                eval_score = self.minimax(new_state, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-Abschneidung
            return min_eval

    def evaluate_state(self, state: GameState) -> int:
        """
        Bewertungsfunktion für den aktuellen GameState.
        """
        return state.points[0] - state.points[1]  # Beispielhafte Bewertung: Punktdifferenz
