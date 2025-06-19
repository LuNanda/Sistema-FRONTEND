# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import umap.umap_ as umap
import hdbscan
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder
print("All imports successful!")

#BOTON DE REESTABLECER BUSQUEDA AVANZADA
if "reset_busqueda" not in st.session_state:
    st.session_state["reset_busqueda"] = False

def buscar_instituciones(nombre):
    url = f"https://api.openalex.org/institutions?search={nombre}"
    resp = requests.get(url)
    if resp.status_code == 200:
        results = resp.json().get("results", [])
        opciones = []
        for r in results:
            opciones.append({
                "id": r["id"].split("/")[-1],  # Solo el ID corto
                "nombre": r["display_name"],
                "pais": r.get("country_code", "").upper()
            })
        return opciones
    return []

st.set_page_config(layout="wide")

# ===== BARRA DE BÚSQUEDA PARA KEYWORDS =====
st.sidebar.title("🔎 Buscar temas")
user_keywords = st.sidebar.text_input("Palabras clave para buscar en OpenAlex:", 
                                      value="CubeSat OR CubeSats OR Nanosatellite")

from datetime import datetime

# ===== BARRA DE BÚSQUEDA AVANZADA=====
st.sidebar.markdown("---")
st.sidebar.title("🔎 Búsqueda Avanzada")

# ===== RESTABLECER BÚSQUEDA AVANZADA=====
if st.sidebar.button("🔄 Restablecer búsqueda avanzada", key="reset_button"):

    for key in ["selected_years", "concept_input", "inst_input", "concept_id", "inst_id"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Años de publicación
current_year = datetime.now().year
years = list(range(current_year, current_year - 10, -1))
selected_years = st.sidebar.multiselect(
    "Años de publicación (últimos 10):",
    years,
    default=[] if st.session_state["reset_busqueda"] else st.session_state.get("selected_years", [])
)
st.session_state["selected_years"] = selected_years

# Tipo de publicación
pub_types = {
    "Todos": "",
    "Artículo de revista": "journal-article",
    "Conferencia": "proceedings-article",
    "Preprint": "posted-content"
}
selected_pub_type = st.sidebar.selectbox(
    "Tipo de publicación:",
    list(pub_types.keys()),
    index=0
)
pub_type_filter = pub_types[selected_pub_type]

# Disciplina científica (concepto OpenAlex)
# Si no está definido aún
if "concept_input" not in st.session_state:
    st.session_state["concept_input"] = ""

concept_input = st.sidebar.text_input(
    "Disciplina científica (opcional):",
    value="" if st.session_state["reset_busqueda"] else st.session_state.get("concept_input", "")
)
st.session_state["concept_input"] = concept_input
concept_id = None
if concept_input:
    def buscar_conceptos_openalex(term):
        url = f"https://api.openalex.org/concepts?search={term}"
        r = requests.get(url)
        if r.status_code != 200:
            return []
        data = r.json()
        return [{"display_name": c["display_name"], "id": c["id"]} for c in data["results"]]

    conceptos = buscar_conceptos_openalex(concept_input)
    if conceptos:
        nombres_conceptos = [c["display_name"] for c in conceptos]
        seleccion = st.sidebar.selectbox("Selecciona una disciplina:", nombres_conceptos)
        concept_id = conceptos[nombres_conceptos.index(seleccion)]["id"]
    else:
        st.sidebar.warning("No se encontraron disciplinas con ese término.")

# Institución (opcional)
inst_input = st.sidebar.text_input(
    "Buscar institución (opcional):",
    value="" if st.session_state["reset_busqueda"] else st.session_state.get("inst_input", "")
)
st.session_state["inst_input"] = inst_input
inst_id = None
if inst_input:
    opciones = buscar_instituciones(inst_input)
    if opciones:
        nombres_mostrados = [f"{o['nombre']} ({o['pais']})" for o in opciones]
        seleccion = st.sidebar.selectbox("Selecciona la institución:", nombres_mostrados)
        inst_id = opciones[nombres_mostrados.index(seleccion)]["id"]
    else:
        st.sidebar.warning("No se encontraron instituciones con ese nombre.")

# ✅ Resetear bandera luego de aplicar
if st.session_state["reset_busqueda"]:
    st.session_state["reset_busqueda"] = False

# Convertimos el keyword a título capitalizado y usamos como parte del título
display_title = user_keywords.title() if user_keywords.strip() else "Scientific"
st.title(f"🔭 {display_title} Concept Clusters in Latin America")

# ===============================
# 🔄 Obtener datos desde OpenAlex
# ===============================
@st.cache_data(show_spinner=True)

def fetch_data(keywords, years, pub_type, institution_id, concept_id):
    base_url = 'https://api.openalex.org/works'
    per_page = 200
    max_pages = 5
    all_results = []

    filters = []

    # Si no hay filtros aplicados, limitar a LATAM
    if not any([institution_id, concept_id]):
        latam_countries = ['HN', 'MX', 'GT', 'SV', 'NI', 'CR', 'PA', 'CU', 'CO', 
                        'VE', 'EC', 'PE', 'DO', 'PR', 'BO', 'PY', 'BR', 'AR', 'UY']
        filters.append(f"authorships.institutions.country_code:{'|'.join(latam_countries)}")

    if years:
        year_filter = '|'.join([str(y) for y in years])
        filters.append(f"publication_year:{year_filter}")
    if pub_type:
        filters.append(f"type:{pub_type}")
    if institution_id:
        filters.append(f"institutions.id:{institution_id}")
    if concept_id:
        filters.append(f"concepts.id:{concept_id}")

    filter_str = ",".join(filters)

    for page in range(1, max_pages + 1):
        url = f"{base_url}?search={keywords}&per-page={per_page}&page={page}&filter={filter_str}"
        response = requests.get(url)
        if response.status_code != 200:
            st.error(f"❌ Error en la página {page} al obtener datos desde OpenAlex.")
            break
        data = response.json()
        if 'results' not in data or not data['results']:
            break
        all_results.extend(data['results'])

    return all_results

# ===== OBTENER DATOS SEGÚN KEYWORDS Y FILTROS AVANZADOS =====
data = fetch_data(
    user_keywords,
    selected_years,
    pub_type_filter,
    inst_id,
    concept_id
)


# Si no hay datos, terminar ejecución
if not data:
    st.warning("No se encontraron resultados con los términos buscados.")
    st.stop()

# ===============================
# 🧠 Procesamiento de conceptos
# ===============================
concept_lists = [[c['display_name'] for c in work.get('concepts', [])] for work in data]
all_concepts = sum(concept_lists, [])
top_concepts_counts = Counter(all_concepts).most_common(100)
top_concepts = [c for c, _ in top_concepts_counts]

# ===============================
# 📊 Vectorización y reducción
# ===============================
vectorizer = CountVectorizer(vocabulary=top_concepts, binary=True)
X = vectorizer.fit_transform(['; '.join(concepts) for concepts in concept_lists])

reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
embedding = reducer.fit_transform(X.T)

# ===============================
# 🔍 Clustering HDBSCAN
# ===============================
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
clusters = clusterer.fit_predict(embedding)

# ===============================
# 📋 Preparar DataFrame
# ===============================
min_len = min(len(top_concepts), len(clusters), len(top_concepts_counts))
df = pd.DataFrame({
    'Concept': top_concepts[:min_len],
    'X': embedding[:min_len, 0],
    'Y': embedding[:min_len, 1],
    'Cluster': clusters[:min_len],
    'Frequency': [count for _, count in top_concepts_counts[:min_len]]
})

df_filtered = df[df['Cluster'] >= 0]
if df_filtered.empty:
    st.error("❌ No se encontraron clusters válidos. Ajusta los parámetros.")
    st.stop()

# ==================================
# 🎛️ Sidebar: selección de cluster
# ==================================
cluster_ids = sorted(df_filtered['Cluster'].unique())
st.sidebar.title("🔎 Explorar Clústers")
selected_cluster = st.sidebar.selectbox("Selecciona un clúster para explorar:", cluster_ids)
df_selected = df_filtered[df_filtered['Cluster'] == selected_cluster]

# ===============================
# 🌐 Red de conceptos
# ===============================
st.subheader("🌐 Cluster Network")

def build_graph(df_filtered, concept_lists):
    G = nx.Graph()
    concept_set = set(df_filtered['Concept'])

    for _, row in df_filtered.iterrows():
        G.add_node(row['Concept'], size=row['Frequency'], cluster=row['Cluster'], pos=(row['X'], row['Y']))

    for concepts in concept_lists:
        filtered = [c for c in concepts if c in concept_set]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                c1, c2 = filtered[i], filtered[j]
                if G.has_edge(c1, c2):
                    G[c1][c2]['weight'] += 1
                else:
                    G.add_edge(c1, c2, weight=1)
    return G

G = build_graph(df_filtered, concept_lists)

if not G.nodes:
    st.warning("⚠️ El grafo está vacío.")
else:
    pos = nx.get_node_attributes(G, 'pos')
    colors = px.colors.qualitative.Set2
    node_colors = [colors[G.nodes[n]['cluster'] % len(colors)] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] * 1.5 for n in G.nodes()]  # 🔽 Tamaño más compacto
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=list(zip(*pos.values()))[0], y=list(zip(*pos.values()))[1],
                            mode='markers+text', text=list(G.nodes),
                            marker=dict(color=node_colors, size=node_sizes, line_width=1))

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        title='Interactive Cluster Network',
        showlegend=False, margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    ))
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================
# 🍯 Honeycomb Cluster Plot (Suave con conceptos por celda)
# ==============================================================
st.subheader("🍯 Honeycomb Cluster Plot")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Estilo suave
sns.set_style("white")

# Crear figura
fig2, ax = plt.subplots(figsize=(12, 8))

# Usar colormap suave
hb = ax.hexbin(
    df_filtered['X'], df_filtered['Y'],
    gridsize=30,
    cmap='YlGnBu',
    mincnt=1
)

# Colorbar
cb = fig2.colorbar(hb, ax=ax)
cb.set_label('Density of Concepts')

# Mostrar textos (opcional si hay poco solapamiento)
for _, row in df_filtered.iterrows():
    ax.text(
        row['X'], row['Y'], row['Concept'],
        fontsize=7.5,
        ha='center', va='center',
        color='black',
        alpha=0.7
    )

ax.axis('off')
ax.set_title("Honeycomb (Hexbin) Plot of CubeSat Concept Clusters", fontsize=14)
st.pyplot(fig2)

# ===============================
# 🔍 Mostrar conceptos por celda
# ===============================
# Obtener los offsets de cada hexágono
hex_centers = hb.get_offsets()
hex_bins = hb.get_array()

# Crear una lista para mapear cada punto a una celda
points = np.column_stack((df_filtered['X'], df_filtered['Y']))
concepts = df_filtered['Concept'].values

# Función de asignación de punto a celda
from scipy.spatial import cKDTree
tree = cKDTree(hex_centers)
_, indices = tree.query(points)

# Crear un diccionario celda -> [conceptos]
from collections import defaultdict
cell_to_concepts = defaultdict(list)
for idx, cell_idx in enumerate(indices):
    cell_to_concepts[cell_idx].append(concepts[idx])

# Crear DataFrame de conceptos agrupados por densidad
cell_data = []
for i, concept_list in cell_to_concepts.items():
    density = len(concept_list)
    cell_data.append({
        "Density": density,
        "Concepts": ", ".join(concept_list)
    })

concepts_df = pd.DataFrame(cell_data).sort_values("Density", ascending=False)

# 💬 Explicación
st.markdown(
    "🔎 **Explicación:** La mayoría de los conceptos están dispersos de forma uniforme en el espacio, "
    "por lo que cada celda hexagonal contiene solo un concepto (densidad = 1). "
    "Solo una celda tiene densidad 2 porque dos conceptos cayeron muy cerca uno del otro espacialmente."
)

# Mostrar conceptos en celdas más densas
st.write("📌 Conceptos en las celdas más densas:")
st.dataframe(concepts_df.head(5), use_container_width=True)

# ===============================
# 🔍 Detalles del Clúster Seleccionado
# ===============================
st.subheader(f"🔍 Exploración del Clúster {selected_cluster}")
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("**📊 Frecuencias de Conceptos:**")
    fig_bar = px.bar(df_selected.sort_values(by="Frequency", ascending=False),
                     x="Frequency", y="Concept", orientation="h",
                     color="Frequency", color_continuous_scale="Viridis")
    fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.markdown("**📋 Tabla de Conceptos:**")
    st.dataframe(df_selected.sort_values(by="Frequency", ascending=False), use_container_width=True)

# ===============================
# 🎯 Evaluación de Precisión del Clúster
# ===============================
from st_aggrid import AgGrid, GridOptionsBuilder

st.subheader("🎯 Evaluación de Precisión del Clúster")

# Preparar datos para validación
validation_data = []
citations_list = []

for work in data:
    conceptos_work = [concept['display_name'] for concept in work.get('concepts', [])]
    if any(c in df_selected['Concept'].values for c in conceptos_work):
        n_citas = work.get('cited_by_count', 0)
        citations_list.append(n_citas)
        doi = work.get('doi', '')
        doi_url = f"https://doi.org/{doi}" if doi else work.get('id', '')
        display_title = work.get('title', 'Sin título')
        validation_data.append({
            "ID": work['id'].split("/")[-1],
            "Título": display_title,
            "DOI": doi,  # <- DOI copiable
            "Conceptos": ", ".join(conceptos_work),
            "Citas": n_citas,
            "Relevancia por Citas": False,
            "Relevante": False
        })

# Calcular mediana de citas para marcar relevancia automática
if citations_list:
    mediana_citas = int(np.median(citations_list))
    for row in validation_data:
        row["Relevancia por Citas"] = row["Citas"] >= mediana_citas

df_validation = pd.DataFrame(validation_data)

if not df_validation.empty:
    st.markdown("### 🔍 Validación Manual de Relevancia")
    st.info("Marca los artículos relevantes para calcular la precisión actual. "
            "Haz **doble clic** sobre el DOI para seleccionarlo y copiarlo fácilmente. "
            "La columna 'Relevancia por Citas' indica si el artículo tiene un número de citas igual o superior a la mediana del clúster.")

    gb = GridOptionsBuilder.from_dataframe(df_validation)
    gb.configure_column("Relevante", editable=True)
    gb.configure_column("DOI", editable=True)
    gb.configure_column("Relevancia por Citas", editable=False)
    gb.configure_column("Citas", editable=False)
    gb.configure_column("Conceptos", editable=False)
    gb.configure_column("ID", editable=False, hide=True)
    grid_options = gb.build()

    grid_response = AgGrid(
        df_validation,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        height=350,
        fit_columns_on_grid_load=True
    )
    updated_data = grid_response['data']
    relevant_count = updated_data['Relevante'].sum()
    precision = relevant_count / len(updated_data) if len(updated_data) > 0 else 0

    st.markdown(f"""
    **📈 Precisión Actual del Cluster:**  
    <span style="color: {'#4CAF50' if precision > 0.7 else '#FF5722'}; font-size: 1.5em;">
    {precision:.2%}</span> ({relevant_count}/{len(updated_data)} artículos relevantes)
    """, unsafe_allow_html=True)
else:
    st.warning("No hay artículos para validar en este cluster.")



# ===============================
# 🌞 Árbol de Conocimiento Real
# ===============================

st.subheader("🌞 Árbol de Conocimiento LACCEI – Panorama General")

# Opciones de filtrado
years_available = list(range(2015, 2025))
selected_year = st.selectbox("Selecciona el año:", years_available[::-1])

# Refiltrar data por año
filtered_data = []
for work in data:
    if 'publication_year' in work and work['publication_year'] == selected_year:
        filtered_data.append(work)

if not filtered_data:
    st.warning("⚠️ No hay publicaciones para el año seleccionado.")
    st.stop()

# Contar publicaciones por institución
institution_counts = Counter()
concept_counts = Counter()
institution_concepts = {}

for work in filtered_data:
    main_concept = work['concepts'][0]['display_name'] if work.get('concepts') else "Unknown"
    concept_counts[main_concept] += 1

    for auth in work.get('authorships', []):
        for inst in auth.get('institutions', []):
            name = inst.get('display_name', 'Unknown')
            institution_counts[name] += 1

            if name not in institution_concepts:
                institution_concepts[name] = Counter()
            institution_concepts[name][main_concept] += 1
# Ordenamiento
sort_option = st.selectbox("Ordenar por:", ["Número de publicaciones", "Alfabético"])
if sort_option == "Número de publicaciones":
    sorted_institutions = sorted(institution_counts.items(), key=lambda x: x[1], reverse=True)
else:
    sorted_institutions = sorted(institution_counts.items(), key=lambda x: x[0])

# Preparar DataFrame para tabla
# Diccionario de códigos de país a nombres completos
country_names = {
    'AR': 'Argentina', 'BO': 'Bolivia', 'BR': 'Brasil', 'CL': 'Chile', 'CO': 'Colombia',
    'CR': 'Costa Rica', 'CU': 'Cuba', 'DO': 'República Dominicana', 'EC': 'Ecuador',
    'GT': 'Guatemala', 'HN': 'Honduras', 'MX': 'México', 'NI': 'Nicaragua',
    'PA': 'Panamá', 'PE': 'Perú', 'PR': 'Puerto Rico', 'PY': 'Paraguay',
    'SV': 'El Salvador', 'UY': 'Uruguay', 'VE': 'Venezuela',
    'DE': 'Alemania', 'US': 'Estados Unidos', 'GB': 'Reino Unido', 'FR': 'Francia',
    'IT': 'Italia', 'ES': 'España', 'CN': 'China', 'JP': 'Japón', 'KR': 'Corea del Sur',
    'IN': 'India', 'CA': 'Canadá', 'AU': 'Australia', 'RU': 'Rusia', 'ZA': 'Sudáfrica',
    'BE': 'Bélgica', 'PT': 'Portugal', 'CH': 'Suiza', 'SE': 'Suecia', 'FI': 'Finlandia',
    'NO': 'Noruega', 'IE': 'Irlanda', 'NL': 'Países Bajos', 'DK': 'Dinamarca',
    'AE': 'Emiratos Árabes Unidos', 'EE': 'Estonia', 'GR': 'Grecia', 'PL': 'Polonia',
    'TR': 'Turquía', 'IL': 'Israel', 'NZ': 'Nueva Zelanda', 'SG': 'Singapur',
    'SA': 'Arabia Saudita', 'EG': 'Egipto', 'TH': 'Tailandia', 'CZ': 'Chequia',
    'RO': 'Rumanía', 'UA': 'Ucrania', 'HU': 'Hungría', 'BG': 'Bulgaria', 'LT': 'Lituania',
    'LV': 'Letonia', 'SK': 'Eslovaquia', 'HR': 'Croacia', 'SI': 'Eslovenia', 'LU': 'Luxemburgo'
    # Agrega más si aparecen otros códigos nuevos
}

institution_country_map = {
    'University of Tübingen': 'Alemania',
    'Université de Paris': 'Francia',
    'University of Oxford': 'Reino Unido',
    'Massachusetts Institute of Technology': 'Estados Unidos',
    # Agregar mas al mapa por la logica y por que no reconoce algunos
}

# Preparar DataFrame para tabla con país
institutions_table = []

for inst, pubs in sorted_institutions:
    top_concept = institution_concepts.get(inst, Counter()).most_common(1)[0][0] if institution_concepts.get(inst) else "Desconocido"
    country_code = None

    # Buscar país en la metadata de los trabajos
    for work in filtered_data:
        for auth in work.get('authorships', []):
            for i in auth.get('institutions', []):
                if i.get('display_name') == inst:
                    raw_code = i.get('country_code')
                    country_code = raw_code.upper() if raw_code else None
                    break
            if country_code:
                break
        if country_code:
            break

    # Determinar el nombre del país
    if country_code:
        country_name = country_names.get(country_code, "Desconocido")
    else:
        # Intentar inferir el país por el nombre de la institución
        country_name = institution_country_map.get(inst, "Desconocido")

    institutions_table.append({
        "Institución": inst,
        "País": country_name,
        "Publicaciones": pubs,
        "Conceptos Top": top_concept
    })


df_institutions = pd.DataFrame(institutions_table)


# Sunburst Data
sunburst_df = pd.DataFrame({
    "Parent": ["LACCEI"] * len(concept_counts),
    "Child": list(concept_counts.keys()),
    "Value": list(concept_counts.values())
})

fig_sunburst = px.sunburst(
    sunburst_df,
    names="Child",
    parents="Parent",
    values="Value",
    title="Árbol de Conocimiento LACCEI – Panorama General",
    color="Value",
    color_continuous_scale="Blues"
)

fig_sunburst.update_layout(margin=dict(t=40, l=0, r=0, b=0))
st.plotly_chart(fig_sunburst, use_container_width=True)

# Mostrar tabla
st.subheader("🏛️ Instituciones Publicando en el Año Seleccionado")
st.dataframe(df_institutions, use_container_width=True)


# ============================================================
# 🏛️ Red de Colaboración entre Instituciones (Coautoría)
# ============================================================
st.subheader("🏛️ Red de Colaboración entre Instituciones (Coautoría)")

inst_pairs = []
inst_counts = Counter()

for work in filtered_data:
    insts_in_work = set()
    for auth in work.get('authorships', []):
        for inst in auth.get('institutions', []):
            name = inst.get('display_name', 'Desconocido')
            if name:
                insts_in_work.add(name)
    for inst in insts_in_work:
        inst_counts[inst] += 1
    insts_in_work = list(insts_in_work)
    for i in range(len(insts_in_work)):
        for j in range(i + 1, len(insts_in_work)):
            pair = tuple(sorted([insts_in_work[i], insts_in_work[j]]))
            inst_pairs.append(pair)

pair_counts = Counter(inst_pairs)

G_inst = nx.Graph()
for inst, count in inst_counts.items():
    G_inst.add_node(inst, size=count)
for (i1, i2), w in pair_counts.items():
    G_inst.add_edge(i1, i2, weight=w)

if len(G_inst.nodes) < 2:
    st.info("No hay suficientes colaboraciones institucionales para mostrar la red.")
else:
    pos = nx.spring_layout(G_inst, seed=42, k=0.8)
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_size = []
    for node in G_inst.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)  # Solo el nombre visible
        node_hover.append(f"{node}<br>{G_inst.nodes[node]['size']} publicaciones")
        node_size.append(G_inst.nodes[node]['size'] * 4 + 10)

    edge_x = []
    edge_y = []
    for edge in G_inst.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,  # Solo nombre
        textposition="bottom center",
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='YlGnBu',
            showscale=True,
            colorbar=dict(title="Publicaciones")
        ),
        hoverinfo='text',
        hovertext=node_hover  # Aquí sí sale publicaciones
    )

    fig_inst = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Red de Colaboración entre Instituciones',
            showlegend=False,
            margin=dict(b=40, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    st.plotly_chart(fig_inst, use_container_width=True)

# ==================================
# 🌎 Red de Colaboración de paises
# ==================================

st.subheader("🌎 Red de Colaboración entre Países (Coautoría)")

country_pairs = []
country_counts = Counter()
for work in filtered_data:
    countries_in_work = set()
    for auth in work.get('authorships', []):
        for inst in auth.get('institutions', []):
            code = inst.get('country_code')
            if code:
                country = country_names.get(code.upper(), code.upper())
                countries_in_work.add(country)
    for c in countries_in_work:
        country_counts[c] += 1
    countries_in_work = list(countries_in_work)
    for i in range(len(countries_in_work)):
        for j in range(i + 1, len(countries_in_work)):
            pair = tuple(sorted([countries_in_work[i], countries_in_work[j]]))
            country_pairs.append(pair)

pair_counts = Counter(country_pairs)

G_countries = nx.Graph()
for country, count in country_counts.items():
    G_countries.add_node(country, size=count)
for (c1, c2), w in pair_counts.items():
    G_countries.add_edge(c1, c2, weight=w)

if len(G_countries.nodes) < 2:
    st.info("No hay suficientes colaboraciones internacionales para mostrar la red.")
else:
    pos = nx.spring_layout(G_countries, seed=42, k=0.8)
    node_x = []
    node_y = []
    node_text = []
    node_hover = []
    node_size = []
    for node in G_countries.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)  # Solo el nombre visible
        node_hover.append(f"{node}<br>{G_countries.nodes[node]['size']} publicaciones")
        node_size.append(G_countries.nodes[node]['size'] * 12)

    edge_x = []
    edge_y = []
    for edge in G_countries.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,  # Solo nombre
        textposition="bottom center",
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Publicaciones")
        ),
        hoverinfo='text',
        hovertext=node_hover  # Aquí sí sale publicaciones
    )

    fig_country = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Red de Colaboración entre Países',
            showlegend=False,
            margin=dict(b=40, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    st.plotly_chart(fig_country, use_container_width=True)

#Exportación de datos a PDF
from fpdf import FPDF
from io import BytesIO
import base64

def limpiar_texto(texto):
    if isinstance(texto, str):
        return texto.encode("latin-1", "ignore").decode("latin-1")
    return texto

def cortar_texto(texto, max_len):
    texto = limpiar_texto(str(texto))
    return texto if len(texto) <= max_len else texto[:max_len - 3] + "..."

class PDFConLogo(FPDF):
    def header(self):
        try:
            # Logo pequeño arriba a la derecha
            self.image("Logo-LACCEI.png", x=170, y=3 , w=25)
        except:
            pass
        self.ln(15)  # Separación después del logo

def generar_pdf(df1, df2, keyword, year):
    pdf = PDFConLogo()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # === Título principal ===
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(0)
    titulo_mayus = f"REPORTE DE ANÁLISIS - {keyword.upper()} ({year})"
    pdf.cell(0, 10, limpiar_texto(titulo_mayus), ln=True, align="C")
    pdf.ln(4)  # Menos separación con respecto a la tabla


    # === Tabla 1: Conceptos ===
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, limpiar_texto("Conceptos del Clúster Seleccionado"), ln=True)

    pdf.set_fill_color(180, 220, 255)
    pdf.set_draw_color(150, 150, 150)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(90, 8, "Concepto", 1, 0, 'C', fill=True)
    pdf.cell(40, 8, "Frecuencia", 1, ln=True, fill=True)

    pdf.set_font("Helvetica", "", 10)
    for _, row in df1.iterrows():
        concepto = cortar_texto(row['Concept'], 40)
        frecuencia = str(row['Frequency'])
        pdf.cell(90, 8, concepto, 1)
        pdf.cell(40, 8, frecuencia, 1, ln=True)

    pdf.ln(10)

    # === Tabla 2: Instituciones ===
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, limpiar_texto("Instituciones Publicando en el Año Seleccionado"), ln=True)

    pdf.set_fill_color(180, 220, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(70, 8, "Institución", 1, 0, 'C', fill=True)
    pdf.cell(30, 8, "País", 1, 0, 'C', fill=True)
    pdf.cell(25, 8, "Publicaciones", 1, 0, 'C', fill=True)
    pdf.cell(60, 8, "Concepto Top", 1, ln=True, fill=True)

    pdf.set_font("Helvetica", "", 9)
    for i, (_, row) in enumerate(df2.iterrows()):
        if i % 2 == 0:
            pdf.set_fill_color(245, 245, 245)
            fill = True
        else:
            fill = False

        institucion = cortar_texto(row['Institución'], 45)
        pais = cortar_texto(row['País'], 20)
        publicaciones = str(row['Publicaciones'])
        concepto = cortar_texto(row['Conceptos Top'], 35)

        pdf.cell(70, 8, institucion, border=1, ln=0, fill=fill)
        pdf.cell(30, 8, pais, border=1, ln=0, fill=fill)
        pdf.cell(25, 8, publicaciones, border=1, ln=0, fill=fill)
        pdf.cell(60, 8, concepto, border=1, ln=1, fill=fill)

    # === Exportar PDF para descarga
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_bytes).decode()
    nombre_archivo = f"reporte_{keyword.lower().replace(' ', '_')}_{year}.pdf"
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{nombre_archivo}">📄 Descargar PDF del Análisis</a>'
    return href

# ✅ Mostrar el botón de descarga final
st.markdown("## 📄 Exportar Análisis en PDF")
st.markdown(
    generar_pdf(
        df_selected.sort_values(by="Frequency", ascending=False),
        df_institutions,
        user_keywords,
        selected_year
    ),
    unsafe_allow_html=True
)
