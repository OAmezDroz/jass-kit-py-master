from jass.game.game_util import *
from jass.game.game_observation import GameObservation
from jass.game.const import *
from jass.game.rule_schieber import RuleSchieber
from jass.agents.agent import Agent
import numpy as np

class AgentNoob(Agent):
    def __init__(self):
        super().__init__()
        # We need a rule object to determine the valid cards
        self._rule = RuleSchieber()

    @staticmethod
    def calculate_trump_selection_score(cards, trump: int) -> int:
        # Point values for cards in trump and non-trump suits
        trump_scores = [11, 4, 3, 20, 10, 14, 0, 0, 0]  # Offsets 0 to 8
        non_trump_scores = [11, 4, 3, 2, 10, 0, 0, 0, 0]
        uneufe_scores = [0, 0, 0, 0, 10, 0, 0, 0, 11]  # For Bottom-Up
        obenabe_scores = non_trump_scores  # For Top-Down

        score = 0
        for card in cards:
            card_color = color_of_card[card]
            card_offset = offset_of_card[card]

            if trump == OBE_ABE:
                # Top-Down: Highest cards win
                score += obenabe_scores[card_offset]
            elif trump == UNE_UFE:
                # Bottom-Up: Lowest cards win
                score += uneufe_scores[card_offset]
            elif card_color == trump:
                # Trump suit
                score += trump_scores[card_offset]
            else:
                # Non-trump suit
                score += non_trump_scores[card_offset]
        return score

    def action_trump(self, obs: GameObservation) -> int:
        """
        Determine trump action for the given observation
        Args:
            obs: the game observation, it must be in a state for trump selection

        Returns:
            selected trump as encoded in jass.game.const or jass.game.const.PUSH
        """
        # Get the cards in hand
        cards = convert_one_hot_encoded_cards_to_int_encoded_list(obs.hand)

        # Calculate scores for each trump option
        trump_options = [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE]
        trump_scores = {}
        for trump in trump_options:
            score = self.calculate_trump_selection_score(cards, trump)
            # Boost score for having more trump cards
            if trump in [DIAMONDS, HEARTS, SPADES, CLUBS]:
                trump_count = sum(1 for card in cards if color_of_card[card] == trump)
                score += trump_count * 10
            trump_scores[trump] = score

        # Select the best trump
        best_trump = max(trump_scores, key=trump_scores.get)
        best_score = trump_scores[best_trump]

        # Threshold for passing (PUSH)
        if best_score < 90 and obs.dealer == obs.player:
            return PUSH

        # Ensure the selected trump is valid
        if best_trump not in [DIAMONDS, HEARTS, SPADES, CLUBS, OBE_ABE, UNE_UFE, PUSH]:
            raise ValueError(f"Illegal trump ({best_trump}) selected")

        return best_trump

    def get_card_rank(self, card: int, trump: int) -> int:
        """
        Assigns a rank to the card based on the trump and game mode.
        Higher rank means stronger card.

        Args:
            card: int-encoded card.
            trump: current trump suit.

        Returns:
            An integer rank of the card.
        """
        card_suit = color_of_card[card]
        card_offset = offset_of_card[card]

        if trump == OBE_ABE:
            # Obenabe: Highest cards win
            rank_order = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}
            rank = rank_order[card_offset]
        elif trump == UNE_UFE:
            # Uneufe: Lowest cards win
            rank_order = {8: 8, 7: 7, 6: 6, 5: 5, 4: 4, 3: 3, 2: 2, 1: 1, 0: 0}
            rank = rank_order[card_offset]
        else:
            # Trump game
            if card_suit == trump:
                # Trump suit
                trump_rank_order = {3: 8, 5: 7, 0: 6, 1: 5, 2: 4, 4: 3, 6: 2, 7: 1, 8: 0}
                rank = trump_rank_order[card_offset]
            else:
                # Non-trump suits
                non_trump_rank_order = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}
                rank = non_trump_rank_order[card_offset]
        return rank

    def get_current_winner(self, trick: np.ndarray, trump: int, num_cards_played: int) -> int:
        """
        Determine the current winning player in an incomplete trick.

        Args:
            trick: The current trick array with int-encoded cards.
            trump: The current trump suit.
            num_cards_played: Number of cards played in the trick.

        Returns:
            Index (0 to num_cards_played - 1) of the player who is currently winning.
        """
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
                # Trump game
                if current_is_trump:
                    if winning_is_trump:
                        # Both are trumps, compare ranks
                        if current_rank > winning_rank:
                            winning_index = i
                            winning_card = current_card
                            winning_rank = current_rank
                    else:
                        # Current card is trump, winning card is not
                        winning_index = i
                        winning_card = current_card
                        winning_rank = current_rank
                        winning_is_trump = True
                else:
                    if winning_is_trump:
                        # Winning card is trump, current is not
                        continue
                    else:
                        # Neither is trump, compare if same suit
                        if current_suit == leading_suit:
                            if current_rank > winning_rank:
                                winning_index = i
                                winning_card = current_card
                                winning_rank = current_rank
            else:
                # Obenabe or Uneufe
                if current_suit == leading_suit:
                    if current_rank > winning_rank:
                        winning_index = i
                        winning_card = current_card
                        winning_rank = current_rank
        return winning_index

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Determine the card to play using enhanced strategies.

        Args:
            obs: the game observation

        Returns:
            the card to play, int encoded as defined in jass.game.const
        """
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        valid_card_indices = np.flatnonzero(valid_cards)

        # Get the current trick and number of cards played
        current_trick = obs.current_trick
        num_cards_played = np.sum(current_trick != -1)

        # Determine the leading suit
        if num_cards_played > 0:
            leading_card = current_trick[0]
            leading_suit = color_of_card[leading_card]
        else:
            leading_suit = None

        # Determine if partner is winning
        partner_is_winning = False
        if num_cards_played >= 1:
            # Determine who is currently winning
            trick = current_trick.copy()
            winning_player_index = self.get_current_winner(trick, obs.trump, num_cards_played)
            # Map the trick positions to player IDs
            trick_players = [(obs.forehand + i) % 4 for i in range(num_cards_played)]
            winning_player = trick_players[winning_player_index]
            if winning_player % 2 == obs.player % 2 and winning_player != obs.player:
                partner_is_winning = True

        # Strategy:
        # Find the cards that can win the trick
        winning_cards = []
        for card in valid_card_indices:
            # Simulate the trick with this card
            trick = current_trick.copy()
            trick[num_cards_played] = card
            num_cards_in_simulated_trick = num_cards_played + 1
            # Calculate who would win if we play this card
            winning_player_index = self.get_current_winner(trick, obs.trump, num_cards_in_simulated_trick)
            trick_players = [(obs.forehand + i) % 4 for i in range(num_cards_in_simulated_trick)]
            winning_player = trick_players[winning_player_index]
            if winning_player == obs.player:
                winning_cards.append(card)

        if winning_cards:
            # If partner is winning, avoid overtaking unless necessary
            if partner_is_winning:
                # Play the lowest valid card to conserve high cards
                card_to_play = min(valid_card_indices, key=lambda c: self.get_card_rank(c, obs.trump))
            else:
                # Play the lowest card that wins
                card_to_play = min(winning_cards, key=lambda c: self.get_card_rank(c, obs.trump))
        else:
            # Cannot win the trick, discard the lowest card
            card_to_play = min(valid_card_indices, key=lambda c: self.get_card_rank(c, obs.trump))

        return card_to_play
