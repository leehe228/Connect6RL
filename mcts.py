"stock mcts implementation"
import sys
import copy
import random
import math
import copy

state_size = 15

MCTS_ITERATIONS = 1000

if len(sys.argv) == 2:
    try:
        MCTS_ITERATIONS = int(sys.argv[1])
    except ValueError:
        print(f'Invalid parameter for mcts iterations, defaulting to {MCTS_ITERATIONS}')

def start_game():
    "Start a game"
    game = Board(15) #15 as a placeholder
    while True:
        print(game)
        entry = tuple(map(int, input('Move: ').replace(' ', '').split(',')))
        game.move(entry[0], entry[1])
        if game.check_win():
            print("I lost :(")
            break
        move = mcts_go(copy.deepcopy(game), 1, stats=True)
        game.move(move[0], move[1])
        if game.check_win():
            print("I won :D")
            break

def mcts_go(current_game, team, iterations=MCTS_ITERATIONS, stats=False):
    "MCTS"
    #Initialize the tree with possible moves and current position
    tree = [Node()] #for general tracking and debugging
    for move in current_game.get_obvious_moves():
        new_node = Node(parent=tree[0], move_to=move)
        tree[0].children.append(new_node)
        tree.append(new_node)

    for _ in range(iterations):
        #iterations
        current_node = tree[0] #origin node, current board.
        while not current_node.is_leaf():
            children_scores = tuple(map(lambda x: x.ucb1(), current_node.children))
            current_node = current_node.children[children_scores.index(max(children_scores))]

        board_updates = 0
        for move in current_node.moves_to:
            current_game.move(move[0], move[1], team)
            board_updates += 1

        #quickly check if the game if is in a terminal state
        do_rollout = True
        rollout_res = current_game.check_win()
        if rollout_res:
            do_rollout = False #the game is already terminal, look no further.

        if not current_node.visits and do_rollout: #==0
            #rollout
            rollout_res = rollout(copy.deepcopy(current_game), team)
        elif current_node.visits and do_rollout:
            #let's go deeper!!!!!!111!!!
            for move in current_game.get_obvious_moves():
                new_node = Node(parent=current_node, move_to=list(move))
                current_node.children.append(new_node)
                tree.append(new_node)
            if not current_node.children:
                rollout_res = 0
            else:
                current_node = current_node.children[0]
                #update board again
                board_updates += 1
                current_game.move(current_node.moves_to[-1][0], current_node.moves_to[-1][1], team)
                #rollout
                rollout_res = rollout(copy.deepcopy(current_game), team)

        #revert board
        for _ in range(board_updates):
            current_game.undo()

        #backpropogate the rollout
        while current_node.parent: #not None. only the top node has None as a parent
            current_node.visits += 1
            current_node.score += rollout_res
            current_node = current_node.parent
        current_node.visits += 1 #for the mother node

    #pick the move with the most visits
    if stats:
        print('Stats for nerds\n' f'Search tree size: {len(tree)}')
    current_node = tree[0]
    visit_map = tuple(map(lambda x: x.visits, current_node.children))
    best_move = visit_map.index(max(visit_map))
    return current_game.get_obvious_moves()[best_move]

def rollout(game, team):
    "Rollout a game"
    max_moves = game.size ** 2
    while game.moves < max_moves:
        check_win = game.check_win()
        if check_win:
            return (check_win * team + 1) // 2
        #make a random move
        while True:
            row = random.randint(0, game.size - 1)
            col = random.randint(0, game.size - 1)
            if (row, col) not in game.move_history:
                game.move(row, col, team)
                break
    return 0.5 #draw


class Board:
    "Board"
    def __init__(self, size):
        self.size = size
        self.move_history = []
        self.moves = 0
        self.__board = [[0 for _ in range(size)] for _ in range(size)]

    def move(self, row, col, piece):
        "Place a piece (-1) piece should take the first turn"
        if not piece:
            piece = (self.moves % 2) * 2 - 1
        if self.__board[row][col] == 0 and (piece == 1 or piece == -1):
            self.move_history.append((row, col))
            self.__board[row][col] = piece
            self.moves += 1
        elif piece != 1 and piece != -1:
            print("The piece should be an integer of 0 or 1.")
        else:
            print("The coordinates on the board are already taken.")

    def undo(self):
        "remove the last placed piece"
        if self.move_history: #is not empty
            self.__board[self.move_history[-1][0]][self.move_history[-1][1]] = 0
            self.move_history.pop()
            self.moves -= 1
        else:
            raise IndexError("No moves have been played.")

    def get(self, row, col):
        "Get a piece at row, col"
        return self.__board[row][col]

    def check_win(self): #cross my fingers and hope everything bloody works
        "check if the game has reached a terminal state"
        if not self.move_history:
            return 0
        latest_move = self.move_history[-1]
        #check horizontal area of last placed piece
        start = latest_move[1] - 5
        if start < 0:
            start = 0
        diag_start_col = start #because we can
        end = latest_move[1] + 6
        if end > self.size:
            end = self.size
        diag_end_col = end #because we can
        for start_ in range(0, end - 5):
            result = sum(self.__board[latest_move[0]][start + start_:start + start_ + 6])
            if result == 5:
                return 1
            if result == -5:
                return -1

        #check the vertical area of the last placed piece
        start = latest_move[0] - 5
        if start < 0:
            start = 0
        diag_start_row = start #because we can
        end = latest_move[0] + 6
        if end > self.size:
            end = self.size
        diag_end_row = end #because we can
        vertical = [self.__board[x][latest_move[1]] for x in range(start, end)]
        for start_ in range(0, end - start - 5):
            result = sum(vertical[start_:start_ + 6])
            if result == 5:
                return 1
            if result == -5:
                return -1

        #check the top left - bottom right diagonal
        start = - min((latest_move[0] - diag_start_row, latest_move[1] - diag_start_col))
        end = min((diag_end_row - latest_move[0], diag_end_col - latest_move[1]))
        diagonal = [self.__board[latest_move[0] + x][latest_move[1] + x]
                    for x in range(start, end)] #tuples perform better than lists
        for start_ in range(0, end - start - 5):
            result = sum(diagonal[start_:start_ + 6])
            if result == 5:
                return 1
            if result == -5:
                return -1

        #check bottom left - top right diagonal
        start = - min((latest_move[1] - diag_start_col, diag_end_row - latest_move[0] - 1))
        end = min((diag_end_col - latest_move[1], latest_move[0] - diag_start_row + 1))
        diagonal = [self.__board[latest_move[0] - x][latest_move[1] + x]
                    for x in range(start, end)]
        for start_ in range(0, end - start - 5):
            result = sum(diagonal[start_:start_ + 6])
            if result == 5:
                return 1
            if result == -5:
                return -1

        return 0

    def get_obvious_moves(self):
        """
        Returns a list of obvious moves
        Obvious spots are empty squares adjacent to an existent piece
        """
        moves = []
        for piece in self.move_history:
            directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                          (1, 0), (1, -1), (0, -1), (-1, -1)]
            for direction in directions:
                if (0 <= (piece[0] + direction[0]) < self.size
                        and 0 <= (piece[1] + direction[1]) < self.size):
                    if not self.__board[piece[0] + direction[0]][piece[1] + direction[1]]:
                        #== 0
                        moves.append((piece[0] + direction[0], piece[1] + direction[1]))
        return list(set(moves))

    def __str__(self):
        return ('\n'.join(' '.join(map(str, x)) for x in self.__board).replace('-1', 'X')
               ).replace('1', 'O').replace('0', ' ')

    
class Node:
    def __init__(self, parent=None, move_to=None):
        self.parent = parent #the object
        if parent and not move_to:
            raise TypeError("A parent is provided with no move_to paramenter.")
        elif parent:
            self.moves_to = copy.deepcopy(self.parent.moves_to)
            self.moves_to.append(move_to)
        else:
            self.moves_to = []
        self.score = 0
        self.visits = 0
        self.children = []

    def is_leaf(self):
        return not bool(self.children)

    def ucb1(self):
        try:
            return self.score / self.visits + 2 * math.sqrt(math.log(self.parent.visits)
                                                            / self.visits)
        except ZeroDivisionError:
            #equivalent to infinity
            #assuming log(parent visits) / visits will not exceed 100
            return 10000