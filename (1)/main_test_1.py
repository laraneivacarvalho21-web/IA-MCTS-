import pandas as pd
from popout_game_1 import PopOut
from mcts_player_1 import get_mcts_move
from mcts_pure_1 import get_mcts_pure_move
from id3_algorithm_1 import ID3DecisionTree

def get_human_move(game):
    moves = game.get_valid_moves()
    while True:
        try:
            print("\nJogadas: ", end="")
            for i, (m_type, col) in enumerate(moves):
                print(f"[{i}] {m_type} {col} ", end=" ")
            
            idx = int(input("\nEscolha o número da jogada: "))
            if 0 <= idx < len(moves):
                return moves[idx]
            print("Índice fora do alcance.")
        except ValueError:
            print("Entrada inválida. Digite um número.")

def get_id3_move(game, tree):
    """Pede à árvore de decisão ID3 para prever a próxima jogada."""
    # Achatar o tabuleiro atual para 42 features
    state_row = {}
    for r in range(game.ROWS):
        for c in range(game.COLS):
            state_row[f'cell_{r}_{c}'] = game.board[r][c]
            
    previsao_str = tree.predict_single(tree.root, state_row)
    valid_moves = game.get_valid_moves()
    
    if previsao_str:
        partes = previsao_str.split('_')
        try:
            jogada_id3 = (partes[0], int(partes[1]))
            if jogada_id3 in valid_moves:
                return jogada_id3
        except:
            pass
            
    # Se a árvore prever algo ilegal ou falhar, faz a primeira jogada legal (Segurança)
    print(" (A Árvore ID3 hesitou e fez uma jogada de segurança)")
    return valid_moves[0] 

def play_game(mode, p1_type, p2_type, tree=None, dif_p1="medio", dif_p2="medio"):
    game = PopOut()
    print("\n" + "="*50)
    print(f" JOGO INICIADO: {p1_type.upper()} vs {p2_type.upper()}")
    print("="*50)

    while True:
        game.print_board()
        
        # Regra 3: Repetição (Verificação simplificada para evitar bloqueios)
        if game.is_repetition_draw():
            if (game.current_player == 1 and p1_type == "humano") or (game.current_player == 2 and p2_type == "humano"):
                if input("Estado repetido 3x. Declarar EMPATE? (s/n): ").lower() == 's':
                    print("\nO Jogo terminou em EMPATE por repetição!")
                    break
        
        # Determinar de quem é a vez e as suas características
        is_p1 = (game.current_player == 1)
        tipo_atual = p1_type if is_p1 else p2_type
        dif_atual = dif_p1 if is_p1 else dif_p2
        
        if tipo_atual == "humano":
            current_move = get_human_move(game)
            
        elif tipo_atual == "mcts":
            print(f"\nO Jogador {game.current_player} (MCTS - {dif_atual}) está a pensar...")
            current_move = get_mcts_move(game, dificuldade=dif_atual)

        elif tipo_atual == "mcts_pure":
            print(f"\nO Jogador {game.current_player} (MCTS Puro) está a pensar...")
            current_move = get_mcts_pure_move(game, iterations=10000)
            
        elif tipo_atual == "id3":
            print(f"\nO Jogador {game.current_player} (Árvore ID3) está a pensar...")
            current_move = get_id3_move(game, tree)
        
        # Executar jogada
        res = game.make_move(*current_move)
        
        if res != 0 or not game.get_valid_moves():
            game.print_board()
            if res == -1 or res == 0: 
                print("\nO Jogo terminou em EMPATE!")
            else: 
                print(f"\n*** VENCEDOR: JOGADOR {res} ({'X' if res == 1 else 'O'}) ***")
            break

def pedir_dificuldade(jogador_nome):
    print(f"\nDificuldade para o {jogador_nome}:")
    print("1 - Fácil | 2 - Médio | 3 - Difícil")
    op = input("Escolha (1/2/3): ")
    return {"1": "facil", "2": "medio", "3": "dificil"}.get(op, "medio")

if __name__ == "__main__":

    print("1 - Humano vs Humano")
    print("2 - Humano vs PC (MCTS com heurísticas)")
    print("3 - Humano vs PC (MCTS Puro - sem heurísticas)")
    print("4 - PC vs PC (MCTS vs MCTS)")
    print("5 - PC vs PC (MCTS Puro vs MCTS Puro)")
    print("6 - PC vs PC (MCTS com heurísticas vs MCTS Puro)")
    print("7 - PC vs PC (MCTS vs Árvore ID3)")

    opcao = input("\nEscolha o modo de jogo (1/2/3/4/5/6/7): ")

    if opcao == "1":
        play_game("humano_vs_humano", "humano", "humano")

    elif opcao == "2":
        dif = pedir_dificuldade("PC MCTS")
        play_game("humano_vs_pc", "humano", "mcts", dif_p2=dif)

    elif opcao == "3":
        play_game("humano_vs_pc_puro", "humano", "mcts_pure")

    elif opcao == "4":
        print("\nConfigurar PCs:")
        dif1 = pedir_dificuldade("MCTS 1 (X)")
        dif2 = pedir_dificuldade("MCTS 2 (O)")
        play_game("pc_vs_pc_mcts", "mcts", "mcts", dif_p1=dif1, dif_p2=dif2)

    elif opcao == "5":
        play_game("pc_vs_pc_puro", "mcts_pure", "mcts_pure")

    elif opcao == "6":
        dif = pedir_dificuldade("MCTS com heurísticas (X)")
        play_game("pc_vs_pc_comparacao", "mcts", "mcts_pure", dif_p1=dif)

    elif opcao == "7":
        dif = pedir_dificuldade("PC MCTS (X)")
        print("\nA carregar e treinar a Árvore de Decisão ID3 para a partida...")
        try:
            # Tenta carregar o dataset para a árvore jogar
            df = pd.read_csv("popout_dataset.csv")
            target_col = 'best_move'
            feature_cols = [col for col in df.columns if col != target_col]

            tree = ID3DecisionTree(max_depth=6) # max_depth previne a árvore de ficar demasiado lenta
            tree.fit(df, target_name=target_col, features=feature_cols)
            print("Árvore ID3 treinada com sucesso!")

            play_game("pc_vs_pc_algoritmos", "mcts", "id3", tree=tree, dif_p1=dif)

        except FileNotFoundError:
            print("\n[ERRO] O ficheiro 'popout_dataset.csv' não foi encontrado.")
            print("Corre primeiro o ficheiro que gera o dataset para poderes jogar contra a Árvore ID3!")
    else:
        print("Opção inválida.")