from jass.game.game_util import *
from jass.game.game_sim import GameSim
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
import numpy as np

class AgentMedium(Agent):
    def __init__(self):
        super().__init__()
        # Initialisierung des Regel-Objekts für gültige Karten
        self._rule = RuleSchieber()

    def havePuurWithFour(hand: np.ndarray) -> np.ndarray:
        result = np.zeros(4, dtype=int)
        for color in range(4):
            color_cards = hand[color_offset[color]:color_offset[color] + 9]
            if color_cards[J_offset] == 1 and color_cards.sum() >= 4:
                result[color] = 1
        return result
    
    def enhanced_trump_selection_score(cards, trump: int) -> int:
        trump_score = [15, 10, 7, 25, 6, 19, 5, 5, 5]
        no_trump_score = [9, 7, 5, 2, 1, 0, 0, 0, 0]
        obenabe_score = [14, 10, 8, 7, 5, 0, 5, 0, 0]
        uneufe_score = [0, 2, 1, 1, 5, 5, 7, 9, 11]
        
        score = 0
        pur_count = 0
        for card in cards:
            card_color = color_of_card[card]
            card_offset = offset_of_card[card]
            if trump == OBE_ABE:
                score += obenabe_score[card_offset]
            elif trump == UNE_UFE:
                score += uneufe_score[card_offset]
            elif card_color == trump:
                score += trump_score[card_offset]
                if card_offset == J_offset:
                    pur_count += 1
            else:
                score += no_trump_score[card_offset]
        
        # Bonus für Puur und Nell
        if pur_count > 0:
            score += pur_count * 10
        return score
        
    def action_trump(self, obs: GameObservation) -> int:
        """
        Bestimme die Trumpfauswahl basierend auf der Hand des Spielers.
        
        Args:
            obs: Die Spielbeobachtung zur Trumpfauswahl.

        Returns:
            Ausgewählter Trumpf als Konstante oder PUSH.
        """
        cards = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        
        # Berechne die Trumpfwerte für jede mögliche Option
        trump_scores = {trump: AgentMedium.enhanced_trump_selection_score(cards, trump) 
                        for trump in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]}
        
        # Bester Trumpf auswählen
        best_trump = max(trump_scores, key=trump_scores.get)
        best_score = trump_scores[best_trump]
        
        # Wenn der Score unter dem Schwellenwert liegt, passe (schiebe)
        if best_score < 68 and obs.forehand == -1:
            return PUSH
        
        # Sicherstellen, dass der gewählte Trumpf gültig ist
        if best_trump not in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE, PUSH]:
            raise ValueError(f"Ungültiger Trumpf {best_trump} ausgewählt")
        
        return best_trump

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Wählt die Karte, die gespielt werden soll, basierend auf der Spielposition und strategischen Regeln.

        Args:
            obs: Die Spielbeobachtung.

        Returns:
            Die zu spielende Karte, int-encoded.
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)
        current_trick = obs.current_trick

        # Fall 1: Vorhandspiel (höchste Karte legen)
        if obs.forehand == 0:
            return valid_card_indices[-1]

        # Fall 2: Mit Farbe folgen, falls möglich
        if current_trick[0] != -1:
            leading_suit = color_of_card[current_trick[0]]
            suit_cards = [card for card in valid_card_indices if color_of_card[card] == leading_suit]
            if suit_cards:
                return suit_cards[0]  # die niedrigste Karte der Farbe spielen
            else:
                # keine Farbe zum Folgen, versuche die niedrigste Karte zu spielen
                return valid_card_indices[0]

        # Fallback: Spiele die niedrigste Karte
        return valid_card_indices[0]
