import math
import pandas as pd
from collections import Counter

class TreeNode:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute  # Nome do atributo usado para dividir o nó
        self.label = label          # Se for folha, guarda a classe final
        self.children = {}          # Dicionário: {valor_do_atributo: TreeNode}

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def _entropy(self, target_col):
        """Calcula a entropia de um conjunto de dados H(S)."""
        elements, counts = pd.Series(target_col).value_counts().index, pd.Series(target_col).value_counts().values
        entropy = 0
        for count in counts:
            probabilidade = count / sum(counts)
            entropy -= probabilidade * math.log2(probabilidade)
        return entropy

    def _information_gain(self, data, split_attribute, target_name):
        """Calcula o Information Gain IG(S, A)."""
        total_entropy = self._entropy(data[target_name])
        
        vals, counts = pd.Series(data[split_attribute]).value_counts().index, pd.Series(data[split_attribute]).value_counts().values
        weighted_entropy = 0
        
        for i in range(len(vals)):
            subset = data[data[split_attribute] == vals[i]]
            probabilidade = counts[i] / sum(counts)
            weighted_entropy += probabilidade * self._entropy(subset[target_name])
            
        return total_entropy - weighted_entropy

    def fit(self, data, target_name, features, depth=0):
        """Constrói a árvore de decisão de forma recursiva."""
        target_values = data[target_name].values
        
        # 1. Se todos os exemplos têm a mesma classe, retorna uma folha
        if len(set(target_values)) == 1:
            return TreeNode(label=target_values[0])
            
        # 2. Se não há mais atributos para dividir ou atingimos a profundidade máxima, retorna a classe maioritária
        if len(features) == 0 or (self.max_depth is not None and depth >= self.max_depth):
            majority_class = Counter(target_values).most_common(1)[0][0]
            return TreeNode(label=majority_class)
            
        # 3. Encontrar o melhor atributo usando Information Gain
        gains = [self._information_gain(data, feature, target_name) for feature in features]
        best_feature_idx = gains.index(max(gains))
        best_feature = features[best_feature_idx]
        
        # 4. Criar o nó e dividir
        node = TreeNode(attribute=best_feature)
        remaining_features = [f for f in features if f != best_feature]
        
        # Para cada valor único do melhor atributo, criar um ramo
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            if len(subset) == 0:
                # Se o subset for vazio, adiciona uma folha com a classe maioritária do nó pai
                majority_class = Counter(target_values).most_common(1)[0][0]
                node.children[value] = TreeNode(label=majority_class)
            else:
                # Chamada recursiva
                node.children[value] = self.fit(subset, target_name, remaining_features, depth + 1)
                
        self.root = node
        return node

    def predict_single(self, node, sample):
        """Prevê a classe de um único exemplo descendo a árvore."""
        if node.label is not None: # Chegámos a uma folha
            return node.label
            
        value = sample[node.attribute]
        if value in node.children:
            return self.predict_single(node.children[value], sample)
        else:
            # Caso o valor não exista na árvore (ex: dados de teste novos), 
            # não conseguimos avançar, podemos retornar um valor por defeito ou None
            return None 

    def predict(self, data):
        """Aplica a árvore a um dataframe."""
        return [self.predict_single(self.root, row) for _, row in data.iterrows()]

    def print_tree(self, node=None, indent=""):
        """Apresenta a árvore visualmente no terminal."""
        if node is None:
            node = self.root
            
        if node.label is not None:
            print(indent + "└── Class:", node.label)
            return
            
        print(indent + "├── Atributo:", node.attribute)
        for value, child_node in node.children.items():
            print(indent + "│   ├── Valor (" + str(value) + "):")
            self.print_tree(child_node, indent + "│   │   ")