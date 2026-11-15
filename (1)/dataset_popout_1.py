import pandas as pd
from popout_game_1 import PopOut
from mcts_player_1 import get_mcts_move
from id3_algorithm_1 import ID3DecisionTree

def generate_dataset(num_games=10, filename="popout_dataset.csv"):
    """Põe o MCTS a jogar sozinho para gerar dados de treino."""
    print(f"A gerar dataset simulando {num_games} jogos. Isto pode demorar...")
    dataset = []
    
    for i in range(num_games):
        game = PopOut()
        while True:
            # Capturar o estado atual (achatar a matriz 6x7 para 42 colunas)
            state_row = {}
            for r in range(game.ROWS):
                for c in range(game.COLS):
                    # Guardamos quem tem a peça: 0(vazio), 1(P1), 2(P2)
                    state_row[f'cell_{r}_{c}'] = game.board[r][c]
            
            # Perguntar ao MCTS qual é a melhor jogada
            move = get_mcts_move(game, dificuldade='medio')
            
            # O target é a jogada (ex: "drop_3" ou "pop_4")
            state_row['best_move'] = f"{move[0]}_{move[1]}"
            dataset.append(state_row)
            
            res = game.make_move(*move)
            if res != 0 or not game.get_valid_moves():
                break # Jogo acabou
                
        print(f"Jogo {i+1}/{num_games} concluído. (Dados gerados: {len(dataset)} pares)")
        
    df = pd.DataFrame(dataset)
    df.to_csv(filename, index=False)
    print(f"Dataset guardado como {filename}!")
    return df

if __name__ == "__main__":
    # 1. GERAR DADOS (se já tiveres o CSV, podes comentar esta linha e usar pd.read_csv)
    df = generate_dataset(num_games=25) 
    
    # 2. PREPARAR DADOS PARA O ID3
    target_col = 'best_move'
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Misturar e dividir os dados (Treino / Teste)
    train_data = df.sample(frac=0.85, random_state=42)
    test_data = df.drop(train_data.index)
    
    # 3. TREINAR A ÁRVORE ID3
    print("\nA treinar a Árvore de Decisão ID3 para jogar PopOut...")
    tree = ID3DecisionTree(max_depth=5) # Limitamos a profundidade para evitar árvores gigantes
    tree.fit(train_data, target_name=target_col, features=feature_cols)
    
    # 4. AVALIAR
    predictions = tree.predict(test_data)
    acertos = sum(1 for p, r in zip(predictions, test_data[target_col]) if p == r)
    total = len(test_data)
    
    print(f"\nResultados do Teste (PopOut): {acertos}/{total} acertos ({(acertos/total)*100:.2f}%)")
    print("NOTA: Num jogo complexo, uma precisão de 20-40% já é normal para uma árvore ID3 simples, pois há muitas jogadas aceitáveis.")
    print("\nEstrutura inicial da Árvore:")
    tree.print_tree()