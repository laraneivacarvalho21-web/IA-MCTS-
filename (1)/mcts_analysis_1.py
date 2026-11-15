import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from popout_game_1 import PopOut
from mcts_player_1 import get_mcts_move

# Configuração visual para parecer um trabalho académico limpo
plt.style.use('ggplot')
sns.set_palette("muted")

def simular_jogo(dif_p1, dif_p2):
    """Corre um jogo invisível entre dois bots e regista as métricas."""
    game = PopOut()
    dados_jogo = []
    
    while True:
        jogador_atual = game.current_player
        dificuldade_atual = dif_p1 if jogador_atual == 1 else dif_p2
        
        # Medir o tempo que o MCTS demora a pensar
        inicio = time.time()
        jogada = get_mcts_move(game, dificuldade=dificuldade_atual)
        fim = time.time()
        
        tempo_decorrido = fim - inicio
        tipo_jogada = jogada[0] # 'drop' ou 'pop'
        
        # Registar as métricas deste turno
        dados_jogo.append({
            'Jogador': f'P{jogador_atual} ({dificuldade_atual})',
            'Dificuldade': dificuldade_atual,
            'Tempo_Segundos': tempo_decorrido,
            'Tipo_Jogada': tipo_jogada
        })
        
        resultado = game.make_move(*jogada)
        
        if resultado != 0 or not game.get_valid_moves():
            vencedor = "Empate" if resultado == -1 or resultado == 0 else f'P{resultado} ({dificuldade_atual if resultado == jogador_atual else (dif_p2 if resultado == 1 else dif_p1)})'
            break
            
    return vencedor, dados_jogo

def correr_torneio():
    confrontos = [('facil', 'medio'), ('medio', 'dificil')]
    jogos_por_confronto = 5 # Aumentar para 10 no relatório final para dados mais robustos
    
    todos_dados_turnos = []
    vitorias = {'facil': 0, 'medio': 0, 'dificil': 0, 'Empate': 0}
    
    print("A iniciar torneio de análise MCTS...")
    
    for dif1, dif2 in confrontos:
        print(f"\nSimulando {dif1.upper()} vs {dif2.upper()} ({jogos_por_confronto} jogos)...")
        for i in range(jogos_por_confronto):
            vencedor, dados = simular_jogo(dif1, dif2)
            todos_dados_turnos.extend(dados)
            
            # Contabilizar vitórias (extrair apenas a dificuldade do nome do vencedor)
            if vencedor == "Empate":
                vitorias['Empate'] += 1
            else:
                dif_vencedora = vencedor.split('(')[1].replace(')', '')
                vitorias[dif_vencedora] += 1
                
            print(f"Jogo {i+1} terminado. Vencedor: {vencedor}")

    df_turnos = pd.DataFrame(todos_dados_turnos)
    gerar_graficos(df_turnos, vitorias)

def gerar_graficos(df, vitorias):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Análise de Desempenho: Algoritmo Monte Carlo Tree Search (PopOut)', fontsize=16, fontweight='bold')

    # Gráfico 1: Taxa de Vitórias
    difs = ['facil', 'medio', 'dificil', 'Empate']
    vals = [vitorias[d] for d in difs]
    axes[0].bar(difs, vals, color=['#ff9999', '#66b3ff', '#99ff99', '#c2c2f0'])
    axes[0].set_title('Total de Vitórias no Torneio')
    axes[0].set_ylabel('Número de Vitórias')
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    # Gráfico 2: Tempo de Resposta por Dificuldade (Boxplot)
    sns.boxplot(ax=axes[1], x='Dificuldade', y='Tempo_Segundos', data=df, order=['facil', 'medio', 'dificil'])
    axes[1].set_title('Custo Computacional (Tempo por Jogada)')
    axes[1].set_ylabel('Segundos por Jogada')
    
    # Gráfico 3: Frequência de Jogadas 'Pop' vs 'Drop'
    # Calcular percentagens para tornar a comparação justa
    contagem_jogadas = df.groupby(['Dificuldade', 'Tipo_Jogada']).size().unstack(fill_value=0)
    contagem_jogadas_pct = contagem_jogadas.div(contagem_jogadas.sum(axis=1), axis=0) * 100
    contagem_jogadas_pct = contagem_jogadas_pct.reindex(['facil', 'medio', 'dificil'])
    
    contagem_jogadas_pct.plot(kind='bar', stacked=True, ax=axes[2], color=['#4daf4a', '#e41a1c'])
    axes[2].set_title('Distribuição de Jogadas (Drop vs Pop)')
    axes[2].set_ylabel('Percentagem (%)')
    axes[2].legend(title='Tipo de Jogada')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig('analise_mcts.png') # Guarda o gráfico como imagem para o teu relatório
    plt.show()

if __name__ == "__main__":
    correr_torneio()