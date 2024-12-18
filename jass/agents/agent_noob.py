from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
import numpy as np

class AgentNoob(Agent):
    def __init__(self):
        super().__init__()
        self._rule = RuleSchieber()
        # Track cards played for better decision making
        self._cards_played = np.zeros(36, dtype=bool)

    def _update_cards_played(self, obs: GameObservation):
        """Update tracking of played cards from observation"""
        self._cards_played.fill(False)
        for i in range(obs.nr_tricks):
            for card in obs.tricks[i]:
                if card != -1:
                    self._cards_played[card] = True
        for card in obs.current_trick:
            if card != -1:
                self._cards_played[card] = True

    def calculate_trump_selection_score(self, cards, trump: int) -> float:
        """Enhanced trump selection scoring that considers card combinations"""
        # Base scores for cards
        trump_scores = {
            'A': 11, 'K': 4, 'Q': 3, 'J': 20, '10': 10, 
            '9': 14, '8': 0, '7': 0, '6': 0
        }
        non_trump_scores = {
            'A': 11, 'K': 4, 'Q': 3, 'J': 2, '10': 10,
            '9': 0, '8': 0, '7': 0, '6': 0
        }
        uneufe_scores = {
            'A': 0, 'K': 2, 'Q': 3, 'J': 2, '10': 10,
            '9': 0, '8': 8, '7': 9, '6': 11
        }
        obenabe_scores = non_trump_scores  # Same as non-trump

        score = 0
        for card in cards:
            card_color = color_of_card[card]
            card_offset = offset_of_card[card]
            
            # Convert offset to card rank
            ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6']
            card_rank = ranks[card_offset]
            
            if trump == OBE_ABE:
                score += obenabe_scores[card_rank]
            elif trump == UNE_UFE:
                score += uneufe_scores[card_rank]
            elif card_color == trump:
                score += trump_scores[card_rank]
                # Bonus for having both J and 9 in trump
                if card_rank == 'J' and any(offset_of_card[c] == Nine_offset and 
                                          color_of_card[c] == trump for c in cards):
                    score += 15
                # Bonus for having three or more trump cards
                trump_count = sum(1 for c in cards if color_of_card[c] == trump)
                if trump_count >= 3:
                    score += trump_count * 5
            else:
                score += non_trump_scores[card_rank]

        return score

    def action_trump(self, obs: GameObservation) -> int:
        """Enhanced trump selection with position-based strategy"""
        cards = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)
        
        # Calculate scores for each trump option
        trump_scores = {}
        for trump in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]:
            score = self.calculate_trump_selection_score(cards, trump)
            
            # Position-based adjustments
            if obs.forehand == -1:  # We're in forehand
                if trump in [DIAMONDS, HEARTS, SPADES, CLUBS]:
                    trump_count = sum(1 for card in cards if color_of_card[card] == trump)
                    if trump_count >= 4:  # Bonus for long suits in forehand
                        score += 10
            else:  # We're in backhand
                # Be more aggressive in backhand
                score += 5
                
            trump_scores[trump] = score

        # Select best trump
        best_trump = max(trump_scores, key=trump_scores.get)
        best_score = trump_scores[best_trump]

        # Higher threshold for pushing in forehand
        push_threshold = 90 if obs.forehand == -1 else 70
        if best_score < push_threshold and obs.forehand == -1:
            return PUSH

        return best_trump

    def get_card_rank(self, card: int, trump: int) -> int:
        """Calculate effective rank of card considering game mode"""
        card_suit = color_of_card[card]
        card_offset = offset_of_card[card]
        
        if trump == OBE_ABE:
            rank_order = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}
            rank = rank_order[card_offset]
        elif trump == UNE_UFE:
            rank_order = {8: 8, 7: 7, 6: 6, 5: 5, 4: 4, 3: 3, 2: 2, 1: 1, 0: 0}
            rank = rank_order[card_offset]
        else:
            if card_suit == trump:
                trump_rank_order = {3: 8, 5: 7, 0: 6, 1: 5, 2: 4, 4: 3, 6: 2, 7: 1, 8: 0}
                rank = trump_rank_order[card_offset]
            else:
                non_trump_rank_order = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}
                rank = non_trump_rank_order[card_offset]
        return rank

    def action_play_card(self, obs: GameObservation) -> int:
        """Enhanced card playing strategy with endgame logic"""
        self._update_cards_played(obs)
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        # Get current trick state
        current_trick = obs.current_trick
        num_cards_played = np.sum(current_trick != -1)

        # Determine if we're in endgame (last 3 tricks)
        is_endgame = obs.nr_tricks >= 6

        # If we're leading the trick
        if num_cards_played == 0:
            return self._play_leading_card(obs, valid_card_indices, is_endgame)
        
        # If we're following
        return self._play_following_card(obs, valid_card_indices, is_endgame)

    def _play_leading_card(self, obs: GameObservation, valid_cards: np.ndarray, is_endgame: bool) -> int:
        """Strategy for leading a trick"""
        if is_endgame:
            # In endgame, lead our highest cards
            return max(valid_cards, key=lambda c: self.get_card_rank(c, obs.trump))
        
        # Count remaining cards in each suit
        remaining_by_suit = [0, 0, 0, 0]
        for card in range(36):
            if not self._cards_played[card]:
                remaining_by_suit[color_of_card[card]] += 1

        # Lead from our longest remaining suit
        our_cards_by_suit = [0, 0, 0, 0]
        for card in valid_cards:
            our_cards_by_suit[color_of_card[card]] += 1

        best_suit = -1
        best_length = -1
        for suit in range(4):
            if our_cards_by_suit[suit] > best_length and remaining_by_suit[suit] > 0:
                best_suit = suit
                best_length = our_cards_by_suit[suit]

        if best_suit != -1:
            # Lead highest card from our longest suit
            suit_cards = [c for c in valid_cards if color_of_card[c] == best_suit]
            return max(suit_cards, key=lambda c: self.get_card_rank(c, obs.trump))

        # Fallback to highest card
        return max(valid_cards, key=lambda c: self.get_card_rank(c, obs.trump))

    def _play_following_card(self, obs: GameObservation, valid_cards: np.ndarray, is_endgame: bool) -> int:
        """Strategy for following to a trick"""
        # Check if partner is winning
        partner_winning = False
        if obs.nr_cards_in_trick >= 1:
            trick = obs.current_trick.copy()
            winning_index = self.get_current_winner(trick, obs.trump, obs.nr_cards_in_trick)
            trick_players = [(obs.trick_first_player[obs.nr_tricks] + i) % 4 for i in range(obs.nr_cards_in_trick)]
            winning_player = trick_players[winning_index]
            if winning_player % 2 == obs.player % 2 and winning_player != obs.player:
                partner_winning = True

        # If partner is winning
        if partner_winning:
            if is_endgame:
                # Play our highest card if we can't win
                return max(valid_cards, key=lambda c: self.get_card_rank(c, obs.trump))
            else:
                # Play our lowest card to save strength
                return min(valid_cards, key=lambda c: self.get_card_rank(c, obs.trump))

        # If we can win the trick
        winning_cards = []
        for card in valid_cards:
            trick = obs.current_trick.copy()
            trick[obs.nr_cards_in_trick] = card
            winning_index = self.get_current_winner(trick, obs.trump, obs.nr_cards_in_trick + 1)
            trick_players = [(obs.trick_first_player[obs.nr_tricks] + i) % 4 
                           for i in range(obs.nr_cards_in_trick + 1)]
            if trick_players[winning_index] == obs.player:
                winning_cards.append(card)

        if winning_cards:
            if is_endgame:
                # Play our highest winning card in endgame
                return max(winning_cards, key=lambda c: self.get_card_rank(c, obs.trump))
            else:
                # Play our lowest winning card otherwise
                return min(winning_cards, key=lambda c: self.get_card_rank(c, obs.trump))

        # If we can't win, play our lowest card
        return min(valid_cards, key=lambda c: self.get_card_rank(c, obs.trump))

    def get_current_winner(self, trick: np.ndarray, trump: int, num_cards_played: int) -> int:
        """Determine current winning player in an incomplete trick"""
        leading_card = trick[0]
        leading_suit = color_of_card[leading_card]

        winning_index = 0
        winning_card = leading_card
        winning_suit = color_of_card[winning_card]
        winning_rank = self.get_card_rank(winning_card, trump)
        winning_is_trump = (winning_suit == trump) if trump in [DIAMONDS, HEARTS, SPADES, CLUBS] else False

        for i in range(1, num_cards_played):
            current_card = trick[i]
            current_suit = color_of_card[current_card]
            current_rank = self.get_card_rank(current_card, trump)
            current_is_trump = (current_suit == trump) if trump in [DIAMONDS, HEARTS, SPADES, CLUBS] else False

            if trump in [DIAMONDS, HEARTS, SPADES, CLUBS]:
                if current_is_trump:
                    if winning_is_trump:
                        if current_rank > winning_rank:
                            winning_index = i
                            winning_card = current_card
                            winning_rank = current_rank
                    else:
                        winning_index = i
                        winning_card = current_card
                        winning_rank = current_rank
                        winning_is_trump = True
                else:
                    if not winning_is_trump and current_suit == leading_suit:
                        if current_rank > winning_rank:
                            winning_index = i
                            winning_card = current_card
                            winning_rank = current_rank
            else:
                if current_suit == leading_suit:
                    if current_rank > winning_rank:
                        winning_index = i
                        winning_card = current_card
                        winning_rank = current_rank
        return winning_index