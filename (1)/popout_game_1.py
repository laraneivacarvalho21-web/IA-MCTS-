class PopOut:
    def __init__(self):
        self.ROWS = 6
        self.COLS = 7
        # 0 = Vazio, 1 = X (Jogador 1), 2 = O (Jogador 2)
        self.board = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.current_player = 1
        self.state_history = {}
        # Quando True, _record_state e a deteção de repetição são desativadas.
        # Os clones usados em rollouts MCTS são marcados com esta flag para
        # evitar herdar / contaminar o histórico do jogo real.
        self._track_history = True
        self._record_state()

    def print_board(self):
        symbols = {0: " . ", 1: " X ", 2: " O "}
        header = "  " + "   ".join([str(i) for i in range(self.COLS)])
        print("\n" + header)
        print(" ┌" + "───┬" * (self.COLS - 1) + "───┐")

        for i, row in enumerate(self.board):
            row_str = " │" + "│".join([symbols[cell] for cell in row]) + "│"
            print(row_str)
            if i < self.ROWS - 1:
                print(" ├" + "───┼" * (self.COLS - 1) + "───┤")

        print(" └" + "───┴" * (self.COLS - 1) + "───┘")
        print(f" Vez do Jogador: {'X' if self.current_player == 1 else 'O'}")

    def get_valid_moves(self):
        """
        Devolve apenas jogadas legais reais (drop e pop).
        A 'draw_declaration' foi removida — empates passam a acontecer
        naturalmente por repetição tripla ou por ausência de jogadas legais.
        """
        moves = []
        for col in range(self.COLS):
            # drop: a coluna não está cheia (topo livre)
            if self.board[0][col] == 0:
                moves.append(('drop', col))
            # pop: a peça do fundo pertence ao jogador atual
            if self.board[self.ROWS - 1][col] == self.current_player:
                moves.append(('pop', col))
        return moves

    def _is_legal_move(self, move_type, col):
        """Valida uma jogada antes de a executar."""
        if not isinstance(col, int) or col < 0 or col >= self.COLS:
            return False
        if move_type == 'drop':
            return self.board[0][col] == 0
        if move_type == 'pop':
            return self.board[self.ROWS - 1][col] == self.current_player
        return False

    def make_move(self, move_type, col):
        """
        Executa uma jogada.
        Retornos:
          0  -> jogo continua
          1  -> Jogador 1 venceu
          2  -> Jogador 2 venceu
         -1  -> empate (sem jogadas legais ou repetição tripla)
        Se a jogada for ilegal, levanta ValueError em vez de
        falhar silenciosamente e trocar o turno.
        """
        if not self._is_legal_move(move_type, col):
            raise ValueError(f"Jogada ilegal: ({move_type}, {col}) "
                             f"para o jogador {self.current_player}.")

        last_player = self.current_player

        if move_type == 'drop':
            for r in range(self.ROWS - 1, -1, -1):
                if self.board[r][col] == 0:
                    self.board[r][col] = self.current_player
                    break
        elif move_type == 'pop':
            for r in range(self.ROWS - 1, 0, -1):
                self.board[r][col] = self.board[r - 1][col]
            self.board[0][col] = 0

        winner = self.check_winner(last_player)
        if winner != 0:
            # Trocamos o jogador na mesma para manter consistência,
            # mas devolvemos o vencedor antes de continuar.
            self.current_player = 3 - self.current_player
            self._record_state()
            return winner

        self.current_player = 3 - self.current_player
        self._record_state()

        # Empate por repetição tripla: aplicado automaticamente
        # (tanto no jogo real como em rollouts, exceto se o tracking
        # estiver desligado num clone).
        if self._track_history and self.is_repetition_draw():
            return -1

        # Empate por ausência de jogadas legais para o próximo jogador.
        if not self.get_valid_moves():
            return -1

        return 0

    def _record_state(self):
        if not self._track_history:
            return
        state_key = (tuple(tuple(row) for row in self.board), self.current_player)
        self.state_history[state_key] = self.state_history.get(state_key, 0) + 1

    def is_repetition_draw(self):
        if not self._track_history:
            return False
        state_key = (tuple(tuple(row) for row in self.board), self.current_player)
        return self.state_history.get(state_key, 0) >= 3

    def top_row_full(self):
        """
        Indica se a linha de topo está cheia.
        NOTA: em Pop Out isto NÃO significa fim de jogo —
        ainda podem existir jogadas 'pop' legais.
        """
        return all(self.board[0][col] != 0 for col in range(self.COLS))

    # Alias mantido por retrocompatibilidade caso outro módulo ainda use
    # is_board_full(). Tem o mesmo comportamento que top_row_full().
    def is_board_full(self):
        return self.top_row_full()

    def check_winner(self, last_player):
        p1 = self._has_four(1)
        p2 = self._has_four(2)
        if p1 and p2:
            return last_player
        if p1:
            return 1
        if p2:
            return 2
        return 0

    def _has_four(self, p):
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if all(self.board[r][c + i] == p for i in range(4)):
                    return True
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if all(self.board[r + i][c] == p for i in range(4)):
                    return True
        # Diagonal \
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if all(self.board[r + i][c + i] == p for i in range(4)):
                    return True
        # Diagonal /
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                if all(self.board[r - i][c + i] == p for i in range(4)):
                    return True
        return False

    def clone(self):
        """
        Clone usado por MCTS. Não rastreia histórico de estados:
        - evita herdar contagens de repetição do jogo real
          (que fariam um rollout terminar prematuramente);
        - evita o custo de hash do tabuleiro a cada ply de simulação.
        """
        new_game = PopOut.__new__(PopOut)  # ignora __init__
        new_game.ROWS = self.ROWS
        new_game.COLS = self.COLS
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        new_game.state_history = {}
        new_game._track_history = False
        return new_game