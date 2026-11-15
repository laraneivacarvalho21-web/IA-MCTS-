import pandas as pd
from id3_algorithm_1 import ID3DecisionTree

def discretize_data(df, columns):
    """Transforma valores numéricos em 3 categorias para o ID3."""
    df_disc = df.copy()
    for col in columns:
        # qcut divide em percentis para garantir distribuição equilibrada
        df_disc[col] = pd.qcut(df[col], q=3, labels=["Baixo", "Médio", "Alto"])
    return df_disc

if __name__ == "__main__":
    print("A carregar e a preparar o dataset Iris...")
    # Carregar o dataset via URL (substitui por ficheiro local se necessário)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data = pd.read_csv(url, header=None, names=col_names)
    
    # 1. Discretizar os dados
    features = col_names[:-1]
    data_disc = discretize_data(data, features)
    
    # 2. Dividir em Treino e Teste (80/20)
    train_data = data_disc.sample(frac=0.8, random_state=42)
    test_data = data_disc.drop(train_data.index)
    
    # 3. Treinar a Árvore ID3
    print("\nA treinar a árvore ID3 (do zero)...")
    tree = ID3DecisionTree(max_depth=4)
    tree.fit(train_data, target_name='class', features=features)
    
    # 4. Avaliar
    predictions = tree.predict(test_data)
    acertos = sum(1 for p, r in zip(predictions, test_data['class']) if p == r)
    total = len(test_data)
    
    print(f"\nResultados do Teste: {acertos}/{total} acertos ({(acertos/total)*100:.2f}%)")
    print("\nVisualização da Árvore:")
    tree.print_tree()