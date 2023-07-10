import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import networkx as nx
from nxviz import CircosPlot
from pyvis.network import Network
from IPython.core.display import display, HTML
import numpy as np
import pingouin as pg
import community.community_louvain as community_louvain
from PIL import Image

st.title('Análise de Relações entre Interesses e Hobbies')
st.markdown("""Para essa análise será utilizado um dataset de Hobbies e Interesses disponível no
[Kaggle](https://www.kaggle.com/code/ankur310794/network-analysis-of-hobbies-interests#network-analysis-hobbies-interests-of-young-people).""")

df = pd.read_csv("https://raw.githubusercontent.com/rodrigogomesrc/analise-de-redes/main/datasets/responses.csv")


st.write("""O dataset possui várias colunas referentes a interesses e hobbies, e também algumas colunas sobre características pessoais. 
Para essa análise, foram retiradas as colunas que não eram referentes a interesses e hobbies.""")

st.write("""O objetivo principal da Análise é identificar quais temas são mais relacionados entre si e dividi-los em grupos.
Sendo assim, procuramos identificar a partir do gosto de um tema, quais outros temas também possam ser interessantes para quem gosta dele.""")

st.write("Após retirar as colunas que não eram referentes a Interesses e Hobbies, temos a seguinte lista de colunas:")

cols = [
"Horror",
"Thriller",
"Comedy",
"Romantic",
"Sci-fi",
"War",
"Fantasy/Fairy tales",
"Animated",
"Documentary",
"Western",
"Dance",
"Folk",
"Country",
"Classical music",
"Pop",
"Rock",
"Metal or Hardrock",
"Punk",
"Hiphop, Rap",
"Reggae, Ska",
"Swing, Jazz",
"Rock n roll",
"Alternative",
"Latino",
"Techno, Trance",
"Opera",
"Action",
"History",
"Psychology",
"Politics",
"Mathematics",
"Physics",
"Internet",
"PC",
"Economy Management",
"Biology",
"Chemistry",
"Reading",
"Geography",
"Foreign languages",
"Medicine",
"Law",
"Cars",
"Art exhibitions",
"Religion",
"Countryside, outdoors",
"Dancing",
"Musical instruments",
"Writing",
"Passive sport",
"Active sport",
"Gardening",
"Celebrities",
"Shopping",
"Science and technology",
"Theatre",
"Fun with friends",
"Adrenaline sports",
"Pets",
"Flying"
]
filtered_df = df.loc[:, cols]
filtered_df.columns

def polychronic_correlation(x, y):

    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    covariance = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))
    correlation = covariance / (std_x * std_y)
    return correlation

st.write("Amostra dos dados: ")

st.table(filtered_df.head())

st.write("""
O dataset representa, para cada coluna um tema e para cada linha uma resposta à pesquisa. As respostas são dadas em uma escala de 1 a 5 sobre
quanto o respondente gosta do tema.
""")

st.write("""
O grau da relação entre dois temas é calculado com base na correlação Policrônica. A correlação Policrônica, que é uma medida de correlação
para variáveis categoricas. É feita a correlação entre o gosto de cada um dos temas com o gosto de todos os outros.
""")

st.write("""
O Dataset é modelado usando grafos da seguinte forma: Os nós representam cada um dos temas e as arestas representam a correlação entre os temas.
Dessa forma, quanto mais grossa é a aresta, maior a relação entre os temas. 
""")

st.write("""
Foi definido um Threshold de 0.2 para a correlação, ou seja, apenas as correlações maiores que 0.2 são consideradas. Caso a correlação seja menor
que 0.2, a aresta não é criada.
""")

st.write("""

Uma possível aplicação prática dessa análise é determinar assuntos que um usuário possa gostar de acordo com outros assuntos que ele gosta.
Essa é uma possível abordagem na construção de um sistema de recomendações.
""")


st.set_option('deprecation.showPyplotGlobalUse', False)


#Calculando as correlações entre os temas
correlations = {}

for col1 in filtered_df.columns:
    for col2 in filtered_df.columns:
        if col1 == col2:
          continue
        key = col1 + "&" + col2
        corr = polychronic_correlation(filtered_df[col1], filtered_df[col2])
        correlations[key] = corr


#Criando o grafo

G = nx.Graph()

G.add_nodes_from(cols)

for key, weight in correlations.items():
    if weight > 0.2:
        music_genre, movie_genre = key.split('&')
        G.add_edge(music_genre, movie_genre, weight=weight)

#Filtrando grafo para remover nós de grau 0 e arestas de laço

G.remove_edges_from(nx.selfloop_edges(G))
degrees = dict(G.degree())
filtered_nodes = [node for node, degree in degrees.items() if degree > 0]
filtered_graph = G.subgraph(filtered_nodes)

## Mostrando o grafo filtrado

st.markdown("## Grafo filtrado")


st.write("""

O grafo foi filtrado para remover todas os nós que não têm arestas (grau 0). Ou seja, todos os nós que não têm uma correlação maior que 0.2
com nenhum dos outros temas.
""")

st.write("""

Após a filtragem, temos a seguinte rede:
""")


pos = nx.spring_layout(filtered_graph, seed=42)
nx.draw(filtered_graph, with_labels=False, node_color='lightblue', edge_color='gray', pos=pos)
nx.draw_networkx_labels(filtered_graph, pos, font_size=6)
st.pyplot()

# Mostrando o histograma de distribuição de grau

st.markdown("## Distribuição de graus")

st.write("""

Após calcularmos o histograma de graus, nós vemos que o valor mais comum de graus para um nó da rede é entre 8 e 10.
""")


degree_sequence = [degree for _, degree in filtered_graph.degree()]
plt.hist(degree_sequence, bins='auto', density=True)
plt.xlabel('Grau')
plt.ylabel('Quantidade')
plt.title('Histograma de distribuição de grau')
st.pyplot()

#criando as comunidades

partition = community_louvain.best_partition(filtered_graph, weight='Weight')

st.write("""

Usando o método de Louvain foram encontradas 5 comunidades na rede. As comunidades para esse Dataset são os grupos de temas relacionados.
Em cada comunidade existe uma grande chance de que um usuário que goste de um dos temas goste, ou possa gostar de outro da mesma comunidade.
""")

st.write("""
Uma aplicação em um sistema de recomendação seria recomendar todos os outros temas da mesma comunidade caso o a pessoa ainda não conheça.
""")

for node in filtered_graph.nodes():
    filtered_graph.nodes[node]['partition'] = partition[node]

edge_weights = np.array([filtered_graph[u][v]['weight'] for u, v in filtered_graph.edges()])
edge_widths = (edge_weights / np.percentile(edge_weights, 98)) * 3


#Plotando circos plot com as comunidades

st.markdown("## Circos Plot com as comunidades encontradas")

st.write("""
Podemos visualizar as comunidades também como um Circos Plot:
""")

plt.figure(figsize=(10, 10))
c = CircosPlot(filtered_graph, node_order='partition', node_color='partition')
st.pyplot()


st.markdown("## Grafo padrão com as comunidades encontradas")

st.write("Podemos visualizar o mesmo grafo gerado destacando as comunidades:")

pos = nx.spring_layout(filtered_graph, seed=42)

plt.figure(figsize=(10, 10))
cmap = mpl.colormaps['viridis']
nx.draw_networkx_nodes(filtered_graph, pos, partition.keys(), node_size=70,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(filtered_graph, pos, alpha=0.3, width=edge_widths)
nx.draw_networkx_labels(filtered_graph, pos, font_size=8)

st.pyplot()


#Plotando a matriz de adjacência
st.markdown("## Matriz de adjacência")

st.write("A matrix de adjacência criada é simétrica por ser um grafo não direcionado.")

adj_matrix = nx.adjacency_matrix(G)
adj_df = pd.DataFrame(adj_matrix.toarray(), index=G.nodes(), columns=G.nodes())
plt_1 = plt.figure(figsize=(15, 15))
plt.imshow(adj_df, cmap='hot', interpolation='nearest')
plt.xticks(range(len(adj_df.columns)), adj_df.columns, rotation='vertical')
plt.yticks(range(len(adj_df.index)), adj_df.index)
plt.colorbar()
st.pyplot()


# Engeinvector centrality

st.markdown("## Visualizando nós por Engeinvector Centrality")

st.write("Para esse dataset, valores altos de sinalizam que um tema, apesar de não ser popular, é ligado a outros temas populares.")

eigenvector_centrality = nx.eigenvector_centrality(filtered_graph)
plt.figure(figsize=(10, 10))
color_map = [eigenvector_centrality[node] for node in filtered_graph.nodes()]

pos = nx.spring_layout(filtered_graph, seed=42)
cmap = mpl.colormaps['rainbow']
nx.draw(filtered_graph, with_labels=False, pos=pos, node_color=color_map, edge_color='gray', cmap=cmap)
nx.draw_networkx_labels(filtered_graph, pos, font_size=6)

sm = cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Eingenvector Centrality')
st.pyplot()

# Degree centrality

st.markdown("## Visualizando nós por Degree Centrality")

st.write("""
O gosto de algo com um alto degree centrality está associado (correlacionado) ao gosto de muitas coisas. 
Baseado na visualização gerada, uma possível inferência é quem gosta de música clássica gosta de muitos tipos de música.
""")

degree_centrality = nx.degree_centrality(filtered_graph)

plt.figure(figsize=(10, 10))

color_map = [degree_centrality[node] for node in filtered_graph.nodes()]

pos = nx.spring_layout(filtered_graph, seed=42)
cmap = mpl.colormaps['rainbow']
nx.draw(filtered_graph, with_labels=False, pos=pos, node_color=color_map, edge_color='gray', cmap=cmap)
nx.draw_networkx_labels(filtered_graph, pos, font_size=6)

sm = cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Degree Centrality')
st.pyplot()

#Closeness centrality

st.markdown("## Visualizando nós por Closeness Centrality")

closeness_centrality = nx.closeness_centrality(filtered_graph)

plt.figure(figsize=(10, 10))

color_map = [closeness_centrality[node] for node in filtered_graph.nodes()]

pos = nx.spring_layout(filtered_graph, seed=42)
cmap = mpl.colormaps['rainbow']
nx.draw(filtered_graph, with_labels=False, pos=pos, node_color=color_map, edge_color='gray', cmap=cmap)
nx.draw_networkx_labels(filtered_graph, pos, font_size=6)

sm = cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Closeness Centrality')
st.pyplot()

#Betweenness centrality

st.markdown("## Visualizando nós por Betweenness Centrality")

st.write("""
Os nós com uma maior betweeness centrality ligam as difererentes comunidades. 
Então se quiser unir duas comunidades, pode focar nesse assunto específico.
""")

betweenness_centrality = nx.betweenness_centrality(filtered_graph)

plt.figure(figsize=(10, 10))

color_map = [betweenness_centrality [node] for node in filtered_graph.nodes()]

pos = nx.spring_layout(filtered_graph, seed=42)
cmap = mpl.colormaps['rainbow']
nx.draw(filtered_graph, with_labels=False, pos=pos, node_color=color_map, edge_color='gray', cmap=cmap)
nx.draw_networkx_labels(filtered_graph, pos, font_size=6)

sm = cm.ScalarMappable(cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Closeness Centrality')
st.pyplot()


#Plotando o grafo com pyviz

st.markdown("## Visualização do Pyviz")

nt = Network('1000px', '1000px', notebook=True,  cdn_resources='in_line', bgcolor="#222222", font_color="white")

for node, attributes in filtered_graph.nodes(data=True):
    nt.add_node(node, label=node)

for node1, node2, attributes in filtered_graph.edges(data=True):
    nt.add_edge(node1, node2, weight=attributes['weight'])

nt.barnes_hut()
nt.show("pyvis.html")

HtmlFile = open("pyvis.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, height = 500)

st.markdown("## Visualização criada com Gephi")

#plota a image gephi.png
image = Image.open('gephi.png')
st.image(image, caption='Visualização de comunidades criada com o Gephi', use_column_width=True)