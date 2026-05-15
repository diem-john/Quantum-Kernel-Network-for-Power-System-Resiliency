import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import LineString
import warnings

warnings.filterwarnings('ignore')

# Custom Modules
from src.mapping import ChiayiMicrogridMapper, QuantumWalkIslandingMapper
from src.qkn import QuantumKernelNetwork
from src.qcp import QuantumConformalPredictor

# --- PAGE CONFIG ---
st.set_page_config(page_title="QKN-QCP-CTQW Microgrid Resilience", layout="wide")
st.title("⚡ Quantum-Enhanced Microgrid Resilience Dashboard")
st.markdown(
    "Interactive POC: Quantum Kernel Networks, Conformal Prediction, and Quantum Walks for Typhoon Risk Modeling in Chiayi, Taiwan.")


# --- HELPER: PLOTLY NETWORK VISUALIZATION ---
def plot_interactive_grid(mapper, title="Microgrid Topology", highlighted_nodes=None, node_colors=None):
    """Converts the NetworkX graph with geographic coordinates into an interactive Plotly map."""
    edge_x, edge_y = [], []
    for edge in mapper.graph.edges():
        x0, y0 = mapper.bus_coords[edge[0]].x, mapper.bus_coords[edge[0]].y
        x1, y1 = mapper.bus_coords[edge[1]].x, mapper.bus_coords[edge[1]].y
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color_list = [], [], [], []
    for node in mapper.graph.nodes():
        node_x.append(mapper.bus_coords[node].x)
        node_y.append(mapper.bus_coords[node].y)
        node_text.append(f"Bus {node}")

        # Color logic for islands or highlights
        if node_colors and node in node_colors:
            node_color_list.append(node_colors[node])
        elif highlighted_nodes and node in highlighted_nodes:
            node_color_list.append('red')
        else:
            node_color_list.append('blue')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', textposition="top center",
        hoverinfo='text', marker=dict(size=12, color=node_color_list, line=dict(width=2, color='white')),
        text=[str(n) for n in mapper.graph.nodes()]
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title, titlefont_size=16, showlegend=False, hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="white"
                    ))
    return fig


# --- SESSION STATE INITIALIZATION ---
if 'mapper' not in st.session_state:
    st.session_state.mapper = ChiayiMicrogridMapper()
    st.session_state.mapper.generate_topology()
if 'qkn_trained' not in st.session_state:
    st.session_state.qkn_trained = False
if 'qcp_calibrated' not in st.session_state:
    st.session_state.qcp_calibrated = False

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Pipeline Controls")
n_samples = st.sidebar.slider("Sample Size (POC Speed)", 50, 500, 150)
qkn_qubits = st.sidebar.selectbox("QKN Qubits", [3, 4, 5], index=0)
qkn_layers = st.sidebar.slider("QKN Entangling Layers", 1, 5, 2)
target_coverage = st.sidebar.slider("QCP Target Coverage (%)", 80, 99, 90) / 100.0

# --- PHASE 1 & 2: MAPPING AND DATA ---
st.header("📍 Phase 1: Geospatial Topology (Chiayi)")
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(plot_interactive_grid(st.session_state.mapper, "Enhanced IEEE 33-Bus System"),
                    use_container_width=True)

with col2:
    st.info(
        "**Geospatial Setup**\n\nThe 33-bus system is mapped to Chiayi coordinates. Coastal exposure increases vulnerability for buses further West. The graph is fully interactive—you can pan and zoom.")

    if st.button("Load & Extract Typhoon Data"):
        with st.spinner("Processing Chronological Data..."):
            # MOCK DATA GENERATOR FOR STREAMLIT SPEED (Replace with real CSV logic from main_experiment.py if desired)
            X = np.random.uniform(-np.pi, np.pi, (n_samples, 3))
            Y = np.random.randint(0, 2, n_samples)
            train_b, cal_b = int(0.6 * n_samples), int(0.8 * n_samples)

            st.session_state.X_train, st.session_state.y_train = X[:train_b], Y[:train_b]
            st.session_state.X_cal, st.session_state.y_cal = X[train_b:cal_b], Y[train_b:cal_b]
            st.session_state.X_test, st.session_state.y_test = X[cal_b:], Y[cal_b:]
            st.success("Data Splitting Complete (Train/Cal/Test)")

# --- PHASE 3: QUANTUM KERNEL NETWORK ---
st.header("🧠 Phase 2: Quantum Kernel Training")
if 'X_train' in st.session_state:
    if st.button("Train Quantum Model (QSVM)"):
        with st.spinner("Executing Quantum Circuits on Simulator..."):
            qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)
            qkn.train_qsvm(st.session_state.X_train, st.session_state.y_train)
            st.session_state.qkn_model = qkn
            st.session_state.qkn_trained = True
            st.success("QSVM Trained using Fidelity Quantum Kernel!")

# --- PHASE 4: QUANTUM CONFORMAL PREDICTION ---
st.header("🛡️ Phase 3: Uncertainty Quantification (QCP)")
if st.session_state.qkn_trained:
    if st.button("Calibrate Conformal Predictor"):
        with st.spinner("Calculating Non-Conformity Scores..."):
            qcp = QuantumConformalPredictor(st.session_state.qkn_model, alpha=(1.0 - target_coverage))
            q_hat = qcp.calibrate(st.session_state.X_cal, st.session_state.y_cal, st.session_state.X_train)
            st.session_state.qcp_model = qcp
            st.session_state.q_hat = q_hat
            st.session_state.qcp_calibrated = True

            # Predict on Test Set
            st.session_state.pred_sets = qcp.predict_sets(st.session_state.X_test, st.session_state.X_train,
                                                          classes=[0, 1])

    if st.session_state.qcp_calibrated:
        st.metric(label="Calculated non-conformity threshold (q_hat)", value=f"{st.session_state.q_hat:.4f}")

        # Analyze Results for Dashboard
        coverage = sum(1 for i, p_set in enumerate(st.session_state.pred_sets) if st.session_state.y_test[i] in p_set)
        ambiguous = sum(1 for p_set in st.session_state.pred_sets if len(p_set) == 2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Target Coverage", f"{target_coverage * 100:.1f}%")
        c2.metric("Empirical Coverage", f"{(coverage / len(st.session_state.y_test)) * 100:.1f}%")
        c3.metric("Ambiguous Alerts (Set=2)", f"{ambiguous}")

# --- PHASE 5: CTQW ISLANDING ---
st.header("🌊 Phase 4: CTQW Post-Disaster Islanding")
if st.session_state.qcp_calibrated:
    st.write(
        "Using the ambiguous alerts from the QCP, we simulate line fractures and use a Continuous-Time Quantum Walk to identify survivable microgrid islands.")

    # Let user interactively break lines
    st.write("Select lines predicted to fail by the QKN:")
    default_breaks = ["(2, 19)", "(6, 26)"]
    all_edges = [str(e) for e in st.session_state.mapper.graph.edges()]
    selected_breaks = st.multiselect("Failed Lines", all_edges, default=default_breaks)

    if st.button("Simulate Quantum Walk Islanding"):
        with st.spinner("Evolving Quantum Hamiltonian..."):
            failed_edges = [eval(e) for e in selected_breaks]
            post_grid = st.session_state.mapper.simulate_typhoon_failures(failed_edges)

            qw_mapper = QuantumWalkIslandingMapper(post_grid)
            zones = qw_mapper.identify_islands()

            # Assign colors to different zones for visualization
            colors = px.colors.qualitative.Plotly
            node_colors = {}
            zone_metrics = []

            for i, (zone_id, data) in enumerate(zones.items()):
                color = colors[i % len(colors)]
                for node in data['nodes']:
                    node_colors[node] = color
                zone_metrics.append({"Zone": zone_id, "Size (Nodes)": data['size'], "Buses": str(data['nodes'])})

            col_map, col_data = st.columns([2, 1])
            with col_map:
                st.plotly_chart(plot_interactive_grid(st.session_state.mapper, "Fractured Microgrid Zones",
                                                      node_colors=node_colors), use_container_width=True)
            with col_data:
                st.dataframe(pd.DataFrame(zone_metrics), hide_index=True)
                st.warning(
                    f"**Action Required:** Dispatch local DERs and battery storage to stabilize {len(zones) - 1} isolated islands.")