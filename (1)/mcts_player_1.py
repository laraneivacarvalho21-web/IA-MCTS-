import math
import random
from collections import defaultdict


DIRECOES = ((0, 1), (1, 0), (1, 1), (1, -1))

# Função auxiliar de "exploração". "Caminha" numa só direção e conta quantas peças seguidas do mesmo tipo há
# Argumentos dados: posição atual, direção do movimento, qual a peça que estamos à procura 
def count_sequence(board, row, col, dr, dc, target, rows, cols):
    n = 0
    r = row + dr
    c = col + dc
    while (0 <= r < rows and 0 <= c < cols and board[r][c] == target):
        n += 1
        r += dr
        c += dc
    # Retorono: quantas peças consecutivas há naquela direção (sem contar peça inicial)
    return n

# Objetivo: encontrar a maior sequência numa determinada direção
def maximum_sequence_length(board, landing, col, target, rows, cols):
    best = 0
    for dr, dc in DIRECOES:
        forward  = count_sequence(board, landing, col,  dr,  dc, target, rows, cols)
        backward = count_sequence(board, landing, col, -dr, -dc, target, rows, cols)
        # Soma do nº peças encontradas p a esquerda e p a direita
        total = forward + backward
        if total > best:
            best = total
    # Retorno do comprimento da maior linha encontrada (continuamos sem contabilizar a peça em questão)
    return best

# Escolha da heurística
def heuristic_choice(game, moves, heur_type):
    # Nível aleatório
    if heur_type == 'random':
        return random.choice(moves)

    board = game.board
    rows = game.ROWS
    cols_count = game.COLS
    # Cálculo de qual a coluna central
    center = cols_count // 2
    current = game.current_player
    opponent_p = 3 - current

    # Definição de pesos que fomos alterando mediante testes com o algoritmo
    # Os índices, por ex., OWN(0, 1, 2, 3) correspondem ao número de peças já existentes naquela sequência
    # Quando já temos um alinhamento de 3 e somamos 1 peça = Vitoria
    if heur_type == 'heavy':
        # Pesos crescentes e exponenciais para valorização quando há 2 em linha e principalmente 3
        # Pequena vantagem para OWN relativamente a OPP para priorizar vencer o jogo logo em vez de ir bloquear o adversário
        OWN  = (0, 6, 30, 200)
        OPP  = (0, 5, 25, 180)
    else:
        # Pesos um pouco mais moderados
        # Permite que o random.choices escolha jogadas sub-ótimas de vez em quando - usamos para niveis de dif. do jogo mais baixos
        OWN  = (0, 3, 10, 60)
        OPP  = (0, 2,  8, 50)

    weights = []
    # Para cada movimento possível, calcular um peso para ele
    # m_type: pop ou drop e col: qual a coluna deste movimento
    for m_type, col in moves:
        if m_type == 'drop':
            dist_to_center = abs(col - center)
            # Pesos maiores ao centro e garantia que o peso mínimo é 1
            w = max(1, 10 - dist_to_center * 2)

            # Inicialização da variável que guarda o local onde a peça vai ficar - simulação
            landing = None
            # Percorrer a coluna de baixo para cima pois as peças caem. rows-1 até 0
            for r in range(rows - 1, -1, -1):
                if board[r][col] == 0:
                    # Assim que encontra um espaço vazio guarda essa linha e para o ciclo de procura
                    landing = r
                    break
            # Se houver espaço
            if landing is not None:
                # Quantas peças nossas ficam alinhadas se jogarmos ali?
                own_aligned = maximum_sequence_length(board, landing, col, current,    rows, cols_count)
                # Quantas peças do adversário ficam alinhadas se jogarmos ali?
                opp_aligned = maximum_sequence_length(board, landing, col, opponent_p, rows, cols_count)
                
                # min apenas como mecanismo de segurança pois os tuples de OWN e OPP só tem 4 posições (0, 1,2, 3). Assim garantimos que no máximo só tenta aceder à pos. 3
                # Objetivo: quanto é que esta jogada me ajuda a ganhar?
                # Por ex.: Se houver own_aligned = 3 então vai a OWN[3] e por isso soma 200 ao peso w
                w += OWN[min(own_aligned, 3)]
                # Quanto é que esta jogada impede o meu adversário de ganhar?
                # Tambem damos notas altas por bloquear o adversário
                w += OPP[min(opp_aligned, 3)]
            
            # Já somamos o bónus do centro, o bónus do ataque e o bónus de defesa
            weights.append(w)
        
        # Desvalorização do pop
        elif m_type == 'pop':
            weights.append(1 if heur_type == 'heavy' else 2)
        else:
            weights.append(1)

    # Entre as escolhas possíveis escolhemos de forma random mas informada pois damos pesos a cada jogada.
    # Mantemos a não privisibilidade do monte carlo que é não determinístico
    # Como random.choices retorna uma lista, utilizamos [0] p/ extrair qual o moviemnto
    return random.choices(moves, weights=weights, k=1)[0]


class MCTSNode:

    '''
    State: Representa o estado do tabuleiro - representação em array
    Parent: É igual a None para o root e igual ao nó do qual cada nó derivou para os restantes nós
    Children: Contem todas as possíveis ações a partir do nó atual
    Move : É None para o Root e para os restantes é igual à ação efetuada pelo pai
    visits: Número de vezes que o nó atual foi visitado
    Results: Dicionário para guardar as vitórias de P1 como 1, P2 como 2 e Empate como -1
    Untried_actions: Lista de todas as possíveis ações
    Heur_type: qual a heurística a usar
    '''
    def __init__(self, state, parent=None, move=None, heur_type='light'):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.results = defaultdict(float)
        self.untried_moves = self.state.get_valid_moves()
        # Para diminuir o bias
        random.shuffle(self.untried_moves)
        self.heur_type = heur_type

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    # Equilíbrio entre a Exploitation e a Exploration
    def best_child(self, c_param=1.414):
        # Quem é o player atual? Será que o monte carlo quer ganhar ou evitar que o oponente ganhe?
        player_to_optimize = self.state.current_player
        # Inicialização a - infinito para garantir que qualquer 1ª jogada avaliada é maior que isso
        best_score = -float('inf')
        # best vai guardar o nó vencedor
        best = None
        # Percorrer todas as jogadas possíveis a partir do nó atual
        for child in self.children:
            # Se houver ainda algum filho não testado começar por esse
            if child.visits == 0:
                return child

            # Qual a taxa de vitórias desta jogada tendo em conta o nr vezes que foi simulada? Quanto maior melhor
            win_rate = child.results[player_to_optimize] / child.visits
            # Exploration: Qual foi a jogada testada menos vezes? Usando a fórmula de UCB
            exploration = c_param * math.sqrt(math.log(self.visits) / child.visits)
            score = win_rate + exploration
            if score > best_score:
                best_score = score
                best = child
        return best

    # Fase 2: EXPANSÃO

    def expand(self):
        move = self.untried_moves.pop(0)
        next_state = self.state.clone()
        # Executa esta mudança na nossa "realidade paralela" pois como ainda estamos a testar, não queremos que altere já no jogo
        next_state.make_move(*move)
        # Criação de um novo nó para guardar esse novo futuro na nossa árvore
        child = MCTSNode(next_state, parent=self, move=move, heur_type=self.heur_type)
        self.children.append(child)
        return child

    # SIMULAÇÃO
    def simulate(self):
        # Se escolhermos este caminho, será que leva a vitória?
        # Para responder vai jogar o resto para ver qual seria o resultado
        
        # Continuar a garantir que não há alterações no tabuleiro verdadeiro
        temp_state = self.state.clone()
        depth = 0
        # "Jogo Imaginário"
        while True:
            valid_moves = temp_state.get_valid_moves()
            if not valid_moves:
                return -1, depth # Empate
            
            # Utilização de heurística para ser melhor que aleatório
            move = heuristic_choice(temp_state, valid_moves, self.heur_type)
            res = temp_state.make_move(*move)
            depth += 1
            if res != 0:
                return res, depth

    # A função recebe o resultado da simulação
    def backpropagate(self, result, playout_depth):
        self.visits += 1
        # Caso a jogada tenha terminado em empate
        if result == -1:
            self.results[1] += 0.5
            self.results[2] += 0.5
        else:
            #Fator de recompensa com desconto
            # playout_depth indica qnts movimentos foram necessários na simulação até à vitória
            discount = 0.95 ** playout_depth
            self.results[result] += discount
        if self.parent:
            self.parent.backpropagate(result, playout_depth)


def count_immediate_wins(state, player):
    """Quantas jogadas em 'state' permitem 'player' ganhar imediatamente?"""
    # Queremos saber quando temos uma vitória garantida ou quando o oponente tem uma vitória logo que iria acabar o jogo se não bloqueássemos
    # Garantir que 'state' tem o player certo a jogar
    if state.current_player != player:
        #De forma a conseguirmos trocar o jogador atual sem realmente "estragar" o jogo verdadeiro
        state = state.clone()
        state.current_player = player
    # Inicializa o contador de vitórias imediatas
    n = 0
    for mv in state.get_valid_moves():
        t = state.clone()
        try:
            # Se a jogada levar a vitória somamos 1 ao contador
            if t.make_move(*mv) == player:
                n += 1
        except ValueError:
            continue
    return n


def opponent_can_force_win(state_after_my_move, opponent, me):
    """
    Quer saber: se eu jogar aqui o que é que o adversário pode fazer?
    Após a minha jogada, o adversário tem alguma resposta que crie uma
    ameaça dupla (ou superior) 

    Devolve True se sim. Esta é a deteção de "ameaça a 2 jogadas".
    """
    for opp_move in state_after_my_move.get_valid_moves():
        t = state_after_my_move.clone()
        try:
            res = t.make_move(*opp_move)
        except ValueError:
            continue
        if res == opponent:
            # Adversário ganha já com esta jogada — também é ameaça forçada,
            # mas através da outra função já teriamos apanhado esta ameaça logo não precisamos repetir
            return True
        if res == -1 or res == me:
            # Empate ou vitória nossa: o adversário não tem ameaça
            continue
        # 'res == 0': jogo continua. É a minha vez de novo
        # Mas estamos a perguntar se o ADVERSÁRIO conseguiu criar ameaça
        # dupla, e a ameaça dupla mede-se pela contagem de vitórias DELE
        # depois da MINHA próxima jogada. Como a próxima a jogar sou eu,
        # eu vou tentar bloquear. Logo a verdadeira pergunta é:
        #
        #   "Existe alguma jogada minha que neutralize TODAS as vitórias
        #    do adversário a partir deste novo estado?"
        #
        # Se não existe, então o adversário forçou a vitória
        my_state = t  # vez de eu jogar
        # Para cada jogada minha, conta quantas vitórias do adversário sobram.
        worst_case = None
        for my_move in my_state.get_valid_moves():
            t2 = my_state.clone()
            try:
                r2 = t2.make_move(*my_move)
            except ValueError:
                continue
            if r2 == me:
                worst_case = 0  # se eu ganhei então significa que a ameaça neutralizada
                break
            opp_wins_after = count_immediate_wins(t2, opponent)
            if worst_case is None or opp_wins_after < worst_case:
                worst_case = opp_wins_after
            if worst_case == 0:
                break
        if worst_case is not None and worst_case > 0:
            # Para esta jogada do adversário, o melhor que conseguimos
            # ainda deixa pelo menos 1 vitória dele no ply seguinte
            # Logo o adversário forçou a vitória
            return True
    return False


def tactical_choice(game_state, valid_moves):
    """
    Visão tática estendida recorrendo a diversas etapas
    Camadas:
      1. tenho vitória imediata?
      2. alguma jogada minha previne vitória adversária imediata?
      3 além disso, alguma jogada minha previne que o adversário FORCE a vitória na jogada seguinte (criação de ameaça dupla)?

    """
    current = game_state.current_player
    opponent = 3 - current

    # 1. Tenho vitória imediata?
    for move in valid_moves:
        test = game_state.clone()
        try:
            if test.make_move(*move) == current:
                return ('forced', move)
        except ValueError:
            continue

    # 2. classifica jogadas pelo nº de vitórias imediatas do
    # adversário que sobram a seguir
    move_threats_1ply = []
    for move in valid_moves:
        my_clone = game_state.clone()
        try:
            result = my_clone.make_move(*move)
        except ValueError:
            continue
        if result == current or result == -1:
            move_threats_1ply.append((move, 0, my_clone, result))
            continue
        threats = count_immediate_wins(my_clone, opponent)
        move_threats_1ply.append((move, threats, my_clone, result))

    if not move_threats_1ply:
        return ('none', None)

    min_1ply = min(t for _, t, _, _ in move_threats_1ply)

    if min_1ply > 0:
        # Adversário tem vitória forçada já — escolher menos mau
        least_bad = [m for m, t, _, _ in move_threats_1ply if t == min_1ply]
        if len(least_bad) == 1:
            return ('forced', least_bad[0])
        return ('safe_set', least_bad)

    # Há jogadas que neutralizam ameaças imediatas: então filtra-las
    safe_1ply = [(m, s, r) for m, t, s, r in move_threats_1ply if t == 0]

    # 3. entre as jogadas seguras a 1 ply, quais permitem que o
    # adversário FORCE vitória no ply seguinte? Eliminá-las
    truly_safe = []
    for move, state_after, result in safe_1ply:
        if result == current or result == -1:
            # Jogada que termina o jogo já — totalmente segura.
            truly_safe.append(move)
            continue
        if not opponent_can_force_win(state_after, opponent, current):
            truly_safe.append(move)

    if len(truly_safe) == 1:
        return ('forced', truly_safe[0])
    if len(truly_safe) > 1:
        return ('safe_set', truly_safe)

    # Todas as jogadas seguras a 1 ply permitem ameaça dupla a 2 plies
    # Não há salvação garantida — devolvemos as 1-ply-safe e o MCTS escolhe.
    safe_moves_only = [m for m, _, _ in safe_1ply]
    if len(safe_moves_only) == 1:
        return ('forced', safe_moves_only[0])
    return ('safe_set', safe_moves_only)


def get_mcts_move(game_state, dificuldade='medio'):
    valid_moves = game_state.get_valid_moves()
    if not valid_moves:
        return None

    is_player_one = (game_state.current_player == 1)
    base_c_param = 1.2 if is_player_one else 1.6

    if dificuldade == 'facil':
        iterations = 150
        c_param = 2.5
        use_tactical_vision = True
        heur_type = 'random'
    elif dificuldade == 'dificil':
        iterations = 10000
        c_param = base_c_param * 0.8
        use_tactical_vision = True
        heur_type = 'heavy'
    else:
        iterations = 1000
        c_param = base_c_param
        use_tactical_vision = True
        heur_type = 'light'

    candidate_moves = valid_moves
    if use_tactical_vision:
        decision, payload = tactical_choice(game_state, valid_moves)
        if decision == 'forced':
            return payload
        elif decision == 'safe_set':
            candidate_moves = payload

    root = MCTSNode(game_state.clone(), heur_type=heur_type)
    if candidate_moves is not valid_moves:
        root.untried_moves = list(candidate_moves)
        random.shuffle(root.untried_moves)

    for _ in range(iterations):
        node = root
        while not node.untried_moves and node.children:
            node = node.best_child(c_param=c_param)
        if node.untried_moves:
            node = node.expand()
        result, playout_depth = node.simulate()
        node.backpropagate(result, playout_depth)

    if not root.children:
        return random.choice(candidate_moves)

    if dificuldade == 'facil':
        sorted_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
        top_children = sorted_children[:min(3, len(sorted_children))]
        return random.choice(top_children).move
    else:
        return max(root.children, key=lambda c: c.visits).move