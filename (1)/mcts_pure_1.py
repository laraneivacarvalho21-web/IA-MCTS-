import math
import random
from collections import defaultdict


class MCTSNodePure:
    """
    Pure MCTS node — no heuristics of any kind.
    Selection uses standard UCT (c = sqrt(2)).
    Simulation uses fully random rollouts.
    Backpropagation uses raw win counts (no discount factor).
    """

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move

        self.children = []
        self.visits = 0
        self.wins = defaultdict(float)  # wins[player] accumulates raw wins
        self.untried_moves = self.state.get_valid_moves()
        random.shuffle(self.untried_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.414):
        player = self.state.current_player
        best_score = -float('inf')
        best = None
        for child in self.children:
            if child.visits == 0:
                return child  # visit unvisited children first
            win_rate = child.wins[player] / child.visits
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = win_rate + exploration
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self):
        move = self.untried_moves.pop(0)
        next_state = self.state.clone()
        next_state.make_move(*move)
        child = MCTSNodePure(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def simulate(self):
        """Fully random rollout — no weights, no lookahead."""
        temp_state = self.state.clone()
        while True:
            valid_moves = temp_state.get_valid_moves()
            if not valid_moves:
                return -1  # draw
            move = random.choice(valid_moves)
            result = temp_state.make_move(*move)
            if result != 0:
                return result

    def backpropagate(self, result):
        self.visits += 1
        if result == -1:
            # draw: credit both players equally
            self.wins[1] += 0.5
            self.wins[2] += 0.5
        else:
            self.wins[result] += 1.0
        if self.parent:
            self.parent.backpropagate(result)


def get_mcts_pure_move(game_state, iterations=1000):
    """
    Pure MCTS with no heuristics.
    - c = sqrt(2) (standard UCT constant)
    - Random rollouts
    - No tactical vision / 1-step lookahead
    - Final move selected by most visits (robust child criterion)
    """
    root = MCTSNodePure(game_state.clone())

    for _ in range(iterations):
        node = root

        # 1. Selection
        while not node.untried_moves and node.children:
            node = node.best_child(c_param=1.414)

        # 2. Expansion
        if node.untried_moves:
            node = node.expand()

        # 3. Simulation
        result = node.simulate()

        # 4. Backpropagation
        node.backpropagate(result)

    # Pick the most-visited child
    return max(root.children, key=lambda c: c.visits).move
