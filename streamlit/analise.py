import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from nxviz import CircosPlot
from pyvis.network import Network
from IPython.core.display import display, HTML

st.title('Visualizando dados com Streamlit')
st.markdown("""Para essa análise será utilizado um dataset de Hobbies e Interesses disponível no
[Kaggle](https://www.kaggle.com/code/ankur310794/network-analysis-of-hobbies-interests#network-analysis-hobbies-interests-of-young-people).""")

df = pd.read_csv("https://raw.githubusercontent.com/rodrigogomesrc/analise-de-redes/main/datasets/responses.csv")

movies = ["Horror", 
          "Thriller", 
          "Comedy",
          "Romantic",
          "Sci-fi", 
          "War",
          "Fantasy/Fairy tales", 
          "Animated", 
          "Documentary", 
          "Western", 
          "Action"]

music = ["Dance", 
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
         "Opera"]

filtered_df = df.loc[:, movies + music]

st.markdown("## Colunas do dataset filtrado")

st.write("após filtrar somente as colunas referentes a filmes e músicas, temos as seguintes colunas:")

st.table(filtered_df.head())

st.write("""
É considerado que alguém gosta de dois gêneros ao mesmo tempo se a nota dada para ambos os gêneros for maior ou igual a 3.
Não foi calculado a relação entre dois gêneros do mesmo tipo. Por exemplo, não foi calculado a relação entre Horror e Thriller, 
pois ambos são do tipo filme.
Dessa forma, o grafo resultante é um grafo bipartido. O que fica evidenciado na matriz de adjacência.
""")

st.write("""
Para cada resposta em que uma pessoa gosta de dois gêneros, é acumulado um ponto para a relação entre esses dois gêneros. O valor é normalizado
e a maior relação fica como sendo 1. É filtrado todos os valores em que a relação seja menor do que 0.7, sobrado na visualização somente as 
relações mais fortes entre os gêneros.
""")


st.markdown("## Matriz de adjacência")


st.set_option('deprecation.showPyplotGlobalUse', False)


#Calculando as relações entre os gêneros de música e filmes
relations = {}

for m in music:
    for mov in movies:
        count = ((filtered_df[m] >= 3) & (filtered_df[mov] >= 3)).sum()
        key = f"{m}&{mov}"
        relations[key] = count

max_count = max(relations.values())
normalized_relations = {key: round(count / max_count, 2) for key, count in relations.items()}

G = nx.Graph()

G.add_nodes_from(movies, ntype='movies')
G.add_nodes_from(music, ntype='music')

for key, weight in normalized_relations.items():
    if weight > 0.7:
        music_genre, movie_genre = key.split('&')
        G.add_edge(music_genre, movie_genre, weight=weight)

#Plotando a matriz de adjacência

adj_matrix = nx.adjacency_matrix(G)
adj_df = pd.DataFrame(adj_matrix.toarray(), index=G.nodes(), columns=G.nodes())

plt.imshow(adj_df, cmap='hot', interpolation='nearest')
plt.xticks(range(len(adj_df.columns)), adj_df.columns, rotation='vertical')
plt.yticks(range(len(adj_df.index)), adj_df.index)
plt.colorbar()
st.pyplot()

st.markdown("## Circus Plot com os gêneros de música e filmes")

#Plotando o grafo com pyviz

for n, d in G.nodes(data=True):
    G.nodes[n]['degree'] = G.degree(n)

c = CircosPlot(G, node_order='degree', node_color='ntype', node_grouping='ntype')
st.pyplot()


nt = Network('1000px', '1000px', notebook=True,  bgcolor="#222222", font_color="white")

for node, attributes in G.nodes(data=True):
    nt.add_node(node, label=node)

for node1, node2, attributes in G.edges(data=True):
    nt.add_edge(node1, node2, weight=attributes['weight'])

nt.barnes_hut()
nt.show("pyvis.html")

HtmlFile = open("pyvis.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()


st.markdown("## Visualização do Pyviz")

components.html(source_code, height = 900,width=900)