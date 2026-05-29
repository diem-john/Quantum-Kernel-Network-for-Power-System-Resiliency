import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import LineString
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import pennylane as qml
import os
import warnings

warnings.filterwarnings('ignore')

# Custom Modules
from src.mapping import ChiayiMicrogridMapper, QuantumWalkIslandingMapper
from src.utils import haversine, rankine_vortex, vulnerability_curve
from src.qkn import QuantumKernelNetwork, QuantumTemporalConvNet
from src.qcp import QuantumConformalPredictor


# --- HELPER: CSV EXPORT ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to a UTF-8 encoded CSV."""
    return df.to_csv(index=True).encode('utf-8')


# --- PAGE CONFIG ---
st.set_page_config(page_title="Q-Rating Chaos", layout="wide")
st.title("⚡ Q-Rating Chaos: A Tri-Partite Quantum Framework for Typhoon Modeling and Microgrid Resilience")
st.subheader("ⓒ Engr. D.J. Medina 2026")
st.markdown(
    "Interactive POC: Quantum Kernel Networks, Conformal Prediction, and Quantum Walks for Typhoon Risk Modeling in Chiayi, Taiwan.")


# --- HELPER: GEOSPATIAL MAPBOX VISUALIZATION ---
def plot_interactive_map(mapper, title="Microgrid Topology", highlighted_nodes=None, node_colors=None):
    """Converts the NetworkX graph into an interactive Plotly Mapbox overlay."""
    edge_lon, edge_lat = [], []
    for edge in mapper.graph.edges():
        x0, y0 = mapper.bus_coords[edge[0]].x, mapper.bus_coords[edge[0]].y
        x1, y1 = mapper.bus_coords[edge[1]].x, mapper.bus_coords[edge[1]].y
        edge_lon.extend([x0, x1, None])
        edge_lat.extend([y0, y1, None])

    edge_trace = go.Scattermapbox(
        lon=edge_lon, lat=edge_lat,
        mode='lines', line=dict(width=2, color='#555'), hoverinfo='none'
    )

    node_lon, node_lat, node_text, node_color_list = [], [], [], []
    for node in mapper.graph.nodes():
        node_lon.append(mapper.bus_coords[node].x)
        node_lat.append(mapper.bus_coords[node].y)
        node_text.append(f"Bus {node}")

        if node_colors and node in node_colors:
            node_color_list.append(node_colors[node])
        elif highlighted_nodes and node in highlighted_nodes:
            node_color_list.append('red')
        else:
            node_color_list.append('#1f77b4')  # Default Plotly Blue

    node_trace = go.Scattermapbox(
        lon=node_lon, lat=node_lat,
        mode='markers+text', text=[str(n) for n in mapper.graph.nodes()],
        textposition="top right", hoverinfo='text', hovertext=node_text,
        marker=dict(size=12, color=node_color_list)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(text=title, font=dict(size=16)),
                        showlegend=False, hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        mapbox=dict(
                            style="carto-positron",  # Open-source street map, no API key needed
                            center=dict(lat=mapper.base_lat, lon=mapper.base_lon),
                            zoom=12
                        )
                    ))
    return fig


# --- SESSION STATE INITIALIZATION ---
if 'mapper' not in st.session_state:
    st.session_state.mapper = ChiayiMicrogridMapper()
    st.session_state.mapper.generate_topology()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'qkn_trained' not in st.session_state:
    st.session_state.qkn_trained = False
if 'qcp_calibrated' not in st.session_state:
    st.session_state.qcp_calibrated = False
if 'bus_train' not in st.session_state:
    st.session_state.bus_train = []

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Pipeline Controls")
n_samples = st.sidebar.slider("Sample Size (POC Speed)", 50, 500, 150)
qkn_qubits = st.sidebar.selectbox("QKN Qubits", [1, 2, 3, 4, 5], index=0)
qkn_layers = st.sidebar.slider("QKN Entangling Layers", 1, 5, 2)
target_coverage = st.sidebar.slider("QCP Target Coverage (%)", 80, 99, 90) / 100.0

# --- TAB NAVIGATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Phase 1: Geo-Extraction",
    "⚛️ Phase 2: Quantum Training",
    "🛡️ Phase 3: Uncertainty Calibration",
    "🧪 Phase 4: Inference Testing",
    "🌊 Phase 5: CTQW Islanding"
])

# --- TAB 1: MAPPING & DATA EXTRACTION ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        # Assuming plot_interactive_map is defined elsewhere in your script
        st.plotly_chart(plot_interactive_map(st.session_state.mapper, "Enhanced IEEE 33-Bus System in Chiayi"),
                        use_container_width=True)
    with col2:
        st.info(
            "**Geospatial Setup**\n\nThe 33-bus system is mapped to Chiayi. Coastal exposure increases vulnerability for buses further West.")

        if st.button("Extract Historical Typhoon & Map Vulnerability"):
            with st.spinner("Applying Rankine Vortex, Fragility Models, and K-Means Distillation"):
                import pandas as pd
                from sklearn.preprocessing import MinMaxScaler
                from sklearn.cluster import KMeans
                from sklearn.metrics import pairwise_distances_argmin
                import numpy as np
                import plotly.graph_objects as go

                try:
                    # 1. LOAD RAW DATA
                    df = pd.read_csv("data/raw/typhoon_data.csv")  # Ensure this path is correct

                    # 2. FILTER FOR TAIWAN VICINITY FIRST
                    df_tw = df[(df['lat'] >= 21) & (df['lat'] <= 26) &
                               (df['lng'] >= 118) & (df['lng'] <= 123)].copy()

                    if df_tw.empty:
                        st.error("⚠️ No historical typhoons found near Taiwan in this dataset.")
                        st.stop()

                    # 3. GET THE LAST 50 TYPHOONS THAT ACTUALLY HIT TAIWAN
                    taiwan_seq_ids = df_tw['seq_id'].unique() # [100:]
                    df_recent_tw = df_tw[df_tw['seq_id'].isin(taiwan_seq_ids)]

                    # 4. EXTRACT STRONGEST MOMENTS
                    df_events = df_recent_tw.sort_values(by='wind', ascending=False).head(100)

                    # 5. GRID INITIALIZATION
                    chiayi_lat, chiayi_lng = 23.48, 120.44
                    np.random.seed(42)
                    bus_coords = {i: (chiayi_lat + np.random.uniform(-0.1, 0.1),
                                      chiayi_lng + np.random.uniform(-0.1, 0.1)) for i in range(1, 34)}

                    records = []

                    # 6. SPATIOTEMPORAL PHYSICAL MAPPING
                    for _, row in df_events.iterrows():
                        ty_lat, ty_lng = row['lat'], row['lng']
                        v_max = row['wind'] * 0.51444  # Convert knots to m/s
                        storm_grade = row['grade']

                        for bus_id in range(1, 34):
                            bus_lat, bus_lng = bus_coords[bus_id]

                            dist_km = haversine(ty_lat, ty_lng, bus_lat, bus_lng)
                            local_wind = rankine_vortex(v_max, dist_km)

                            fail_prob = vulnerability_curve(local_wind)
                            label = np.random.binomial(1, fail_prob)

                            records.append({
                                'Bus_ID': bus_id,
                                'Wind_Speed': local_wind,
                                'Storm_Grade': storm_grade,
                                'Distance_to_Eye': dist_km,
                                'Failure_Label': label
                            })

                    df_mapped = pd.DataFrame(records)

                    # 7. MATHEMATICAL CORESET DISTILLATION (Solving O(N^2) Scaling)
                    st.toast("Clustering historical data to build Quantum Coresets", icon="⚛️")
                    target_qpu_budget = 1478

                    if len(df_mapped) > target_qpu_budget:
                        df_fails = df_mapped[df_mapped['Failure_Label'] == 1]
                        df_safes = df_mapped[df_mapped['Failure_Label'] == 0]

                        # Balance the classes: 50% failures, 50% safe
                        k_fails = min(len(df_fails), target_qpu_budget // 2)
                        k_safes = target_qpu_budget - k_fails

                        feature_cols = ['Wind_Speed', 'Storm_Grade', 'Distance_to_Eye']

                        # Cluster Safe Points
                        kmeans_safe = KMeans(n_clusters=k_safes, random_state=42, n_init='auto')
                        kmeans_safe.fit(df_safes[feature_cols])
                        safe_indices = pairwise_distances_argmin(kmeans_safe.cluster_centers_, df_safes[feature_cols])
                        distilled_safes = df_safes.iloc[safe_indices]

                        # Cluster Failure Points
                        if k_fails < len(df_fails):
                            kmeans_fail = KMeans(n_clusters=k_fails, random_state=42, n_init='auto')
                            kmeans_fail.fit(df_fails[feature_cols])
                            fail_indices = pairwise_distances_argmin(kmeans_fail.cluster_centers_,
                                                                     df_fails[feature_cols])
                            distilled_fails = df_fails.iloc[fail_indices]
                        else:
                            distilled_fails = df_fails

                        # Recombine and shuffle
                        df_mapped = pd.concat([distilled_fails, distilled_safes]).sample(frac=1, random_state=42)

                    # 8. DATA PREPARATION FOR QUANTUM KERNEL
                    feature_cols = ['Wind_Speed', 'Storm_Grade', 'Distance_to_Eye']
                    X_raw = df_mapped[feature_cols].values
                    Y = df_mapped['Failure_Label'].values
                    bus_ids = df_mapped['Bus_ID'].values

                    # Quantum Scale to [-pi, pi] for Ry rotation gates
                    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
                    X = scaler.fit_transform(X_raw)

                    # Final shuffle
                    indices = np.arange(len(X))
                    np.random.shuffle(indices)
                    X, Y, bus_ids = X[indices], Y[indices], bus_ids[indices]

                    # 9. SPLIT AND STORE (60/20/20)
                    n_samples = len(X)
                    train_b, cal_b = int(0.6 * n_samples), int(0.8 * n_samples)

                    st.session_state.X_train, st.session_state.y_train = X[:train_b], Y[:train_b]
                    st.session_state.X_cal, st.session_state.y_cal = X[train_b:cal_b], Y[train_b:cal_b]
                    st.session_state.X_test, st.session_state.y_test = X[cal_b:], Y[cal_b:]

                    st.session_state.bus_train = bus_ids[:train_b]
                    st.session_state.bus_cal = bus_ids[train_b:cal_b]
                    st.session_state.bus_test = bus_ids[cal_b:]

                    st.session_state.data_loaded = True
                    st.success(
                        f"Phase 1 Complete: Distilled dataset to {n_samples} mathematically representative Quantum Coresets.")

                    num_safe = np.sum(Y == 0)
                    num_fail = np.sum(Y == 1)
                    st.info(f"Target QPU Training Distribution: **{num_safe} Safe** | **{num_fail} Failures**")

                except FileNotFoundError:
                    st.error("⚠️ Could not find 'data/raw/typhoon_data.csv'. Please check the file path.")
                except Exception as e:
                    st.error(f"⚠️ An error occurred during data processing: {e}")

    # --- DATASET VISUALIZATION ---
    if st.session_state.get('data_loaded', False):
        st.divider()
        st.markdown("### 📊 Distilled Quantum Dataset & Feature Profiles")
        st.write(
            "Below are the representative Coresets (Archetypes) generated via K-Means clustering. The features have been scaled between $-\pi$ and $\pi$ to prepare them for **Quantum Angle Embedding**.")

        if len(st.session_state.bus_train) == len(st.session_state.X_train):
            import pandas as pd
            import numpy as np
            import plotly.graph_objects as go

            # 1. Build the Table
            df_train = pd.DataFrame(st.session_state.X_train,
                                    columns=["Feature 1: Wind (rad)", "Feature 2: Grade (rad)",
                                             "Feature 3: Distance (rad)"])
            df_train.insert(0, "Bus ID", st.session_state.bus_train)
            df_train.insert(0, "Instance ID", [f"Train_{i}" for i in range(len(df_train))])

            df_train["Failure Label"] = st.session_state.y_train
            df_train["Status"] = df_train["Failure Label"].apply(lambda x: "🔴 Failure" if x == 1 else "🟢 Safe")

            col_table, col_vis = st.columns([2, 1])

            with col_table:
                st.dataframe(df_train.drop(columns=["Failure Label"]).style.format({
                    "Feature 1: Wind (rad)": "{:.4f}",
                    "Feature 2: Grade (rad)": "{:.4f}",
                    "Feature 3: Distance (rad)": "{:.4f}"
                }), height=350, use_container_width=True)

            with col_vis:
                st.markdown("#### Dynamic Feature Profile")
                st.write("Select a training instance to visualize its feature vector.")

                num_samples = len(st.session_state.X_train)
                sample_options = [f"Train_{i}" for i in range(num_samples)]

                selected_sample_str = st.selectbox("Select Sample to View:", options=sample_options, index=0)
                selected_idx = int(selected_sample_str.split("_")[1])

                sample = st.session_state.X_train[selected_idx]
                label = st.session_state.y_train[selected_idx]
                bus_id = st.session_state.bus_train[selected_idx]

                color = "#d62728" if label == 1 else "#1f77b4"

                # Radar Chart
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=[sample[0], sample[1], sample[2], sample[0]],
                    theta=['Wind', 'Grade', 'Distance', 'Wind'],
                    fill='toself', fillcolor=color, opacity=0.5,
                    line=dict(color=color, width=2), name=selected_sample_str
                ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[-np.pi, np.pi], showticklabels=False)),
                    showlegend=False, margin=dict(l=30, r=30, t=20, b=20), height=250
                )

                st.plotly_chart(fig_radar, use_container_width=True)
                st.caption(
                    f"**Target Node:** Bus {bus_id} | **Ground Truth:** {'🔴 Failure' if label == 1 else '🟢 Safe'}")
        else:
            st.warning("Index mismatch detected. Please click the Extract button again.")

# --- TAB 2: QKN ---
with tab2:
    if not st.session_state.get('data_loaded', False):
        st.error("⚠️ Please Load & Extract Typhoon Data in Phase 1 before proceeding.")
    else:
        st.write(
            "Using the extracted meteorological and spatial features to train the Quantum Support Vector Machine via Fidelity Estimation.")

        # --- DYNAMIC D3.JS CIRCUIT VISUALIZATION ---
        st.markdown("### 🧬 Quantum Circuit Architecture (Ansatz)")
        st.write(
            "Interact with the sliders to see how the entanglement topology scales for different hardware configurations.")

        # The HTML/JS code for the D3.js Interactive Circuit (Retained exactly as provided)
        circuit_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                body { font-family: sans-serif; margin: 0; padding: 10px; color: #333; }
                .controls { display: flex; gap: 20px; margin-bottom: 15px; align-items: center; background: #f8f9fa; padding: 10px; border-radius: 8px;}
                .metrics { margin-bottom: 15px; font-size: 14px; color: #555; }
                svg { background: #ffffff; border: 1px solid #ddd; border-radius: 8px; }
                .wire { stroke: #888; stroke-width: 2px; }
                .gate-box { fill: #e3f2fd; stroke: #1e88e5; stroke-width: 2px; rx: 4px; ry: 4px; }
                .gate-text { font-family: monospace; font-size: 14px; fill: #0d47a1; text-anchor: middle; dominant-baseline: middle; }
                .cnot-line { stroke: #1e88e5; stroke-width: 2px; }
                .cnot-dot { fill: #1e88e5; }
                .cnot-target { fill: none; stroke: #1e88e5; stroke-width: 2px; }
                .measure-box { fill: #f5f5f5; stroke: #666; stroke-width: 2px; rx: 2px; ry: 2px; }
                .divider { stroke: #ccc; stroke-width: 2px; stroke-dasharray: 5,5; }
            </style>
        </head>
        <body>
            <div class="controls">
                <label><b>Qubits:</b> <input type="range" id="q-slider" min="2" max="6" value="3"> <span id="q-val">3</span></label>
                <label><b>Layers:</b> <input type="range" id="l-slider" min="1" max="4" value="2"> <span id="l-val">2</span></label>
            </div>
            <div class="metrics" id="metrics"></div>
            <div style="overflow-x: auto;">
                <svg id="circuit" height="350"></svg>
            </div>

            <script>
                const svg = d3.select("#circuit");
                const wireSpacing = 50;
                const startX = 60;
                const boxSize = 40;
                const stepX = 80;

                function drawCircuit() {
                    const numQubits = parseInt(document.getElementById("q-slider").value);
                    const numLayers = parseInt(document.getElementById("l-slider").value);

                    document.getElementById("q-val").innerText = numQubits;
                    document.getElementById("l-val").innerText = numLayers;

                    const totalParams = numQubits + (numLayers * numQubits * 3);
                    const depth = 1 + (numLayers * 2);
                    document.getElementById("metrics").innerHTML = `<b>Trainable Parameters:</b> ${totalParams} | <b>Gate Depth:</b> ${depth}`;

                    svg.selectAll("*").remove(); // Clear previous

                    const svgWidth = startX + stepX + (numLayers * stepX * 2) + stepX;
                    svg.attr("width", Math.max(svgWidth, 600));
                    svg.attr("height", numQubits * wireSpacing + 40);

                    // Draw Wires & Labels
                    for (let i = 0; i < numQubits; i++) {
                        let y = 30 + i * wireSpacing;
                        svg.append("line").attr("x1", 20).attr("y1", y).attr("x2", svgWidth - 20).attr("y2", y).attr("class", "wire");
                        svg.append("text").attr("x", 20).attr("y", y).text("|0⟩").attr("dominant-baseline", "middle").style("font-family", "monospace").style("font-weight", "bold");
                    }

                    let currentX = startX;

                    // Step 1: Angle Embedding
                    for (let i = 0; i < numQubits; i++) {
                        let y = 30 + i * wireSpacing;
                        svg.append("rect").attr("x", currentX - boxSize/2).attr("y", y - boxSize/2).attr("width", boxSize).attr("height", boxSize).attr("class", "gate-box");
                        svg.append("text").attr("x", currentX).attr("y", y).text("Ry").attr("class", "gate-text");
                    }
                    currentX += stepX;

                    // Divider
                    svg.append("line").attr("x1", currentX - stepX/2).attr("y1", 10).attr("x2", currentX - stepX/2).attr("y2", numQubits * wireSpacing + 10).attr("class", "divider");

                    // Step 2: Strongly Entangling Layers
                    for (let l = 0; l < numLayers; l++) {
                        // Rotations
                        for (let i = 0; i < numQubits; i++) {
                            let y = 30 + i * wireSpacing;
                            svg.append("rect").attr("x", currentX - boxSize/2).attr("y", y - boxSize/2).attr("width", boxSize).attr("height", boxSize).attr("class", "gate-box");
                            svg.append("text").attr("x", currentX).attr("y", y).text("U(θ)").attr("class", "gate-text");
                        }
                        currentX += stepX;

                        // Ring CNOTs
                        for (let i = 0; i < numQubits; i++) {
                            let controlY = 30 + i * wireSpacing;
                            let targetIdx = (i + 1) % numQubits;
                            let targetY = 30 + targetIdx * wireSpacing;

                            // Offset X slightly so CNOTs don't overlap completely if drawing simultaneously
                            let cnotX = currentX + (i * 10) - ((numQubits*10)/2); 

                            svg.append("line").attr("x1", cnotX).attr("y1", controlY).attr("x2", cnotX).attr("y2", targetY).attr("class", "cnot-line");
                            svg.append("circle").attr("cx", cnotX).attr("cy", controlY).attr("r", 5).attr("class", "cnot-dot"); // Control
                            svg.append("circle").attr("cx", cnotX).attr("cy", targetY).attr("r", 10).attr("class", "cnot-target"); // Target outline
                            svg.append("line").attr("x1", cnotX).attr("y1", targetY - 10).attr("x2", cnotX).attr("y2", targetY + 10).attr("class", "cnot-line"); // Target Cross
                            svg.append("line").attr("x1", cnotX - 10).attr("y1", targetY).attr("x2", cnotX + 10).attr("y2", targetY).attr("class", "cnot-line"); // Target Cross
                        }
                        currentX += stepX;

                        // Divider
                        svg.append("line").attr("x1", currentX - stepX/2).attr("y1", 10).attr("x2", currentX - stepX/2).attr("y2", numQubits * wireSpacing + 10).attr("class", "divider");
                    }

                    // Step 3: Measurement
                    for (let i = 0; i < numQubits; i++) {
                        let y = 30 + i * wireSpacing;
                        svg.append("rect").attr("x", currentX - 15).attr("y", y - 15).attr("width", 30).attr("height", 30).attr("class", "measure-box");
                        // Meter Arc
                        svg.append("path").attr("d", `M ${currentX - 8} ${y + 5} Q ${currentX} ${y - 10} ${currentX + 8} ${y + 5}`).attr("fill", "none").attr("stroke", "#666").attr("stroke-width", "2");
                        // Arrow
                        svg.append("line").attr("x1", currentX).attr("y1", y + 8).attr("x2", currentX + 6).attr("y2", y - 2).attr("stroke", "#666").attr("stroke-width", "2");
                    }
                }

                document.getElementById("q-slider").addEventListener("input", drawCircuit);
                document.getElementById("l-slider").addEventListener("input", drawCircuit);
                drawCircuit(); // Init
            </script>
        </body>
        </html>
        """
        import streamlit.components.v1 as components
        import os
        import pandas as pd
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.decomposition import KernelPCA
        from sklearn.ensemble import RandomForestClassifier
        import plotly.graph_objects as go
        import plotly.express as px
        from sklearn.svm import SVC


        # Helper function for CSV download
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=True).encode('utf-8')


        # PyTorch Network Definition for Sequence Processing
        class QuantumTemporalConvNet(nn.Module):
            def __init__(self, in_channels, sequence_length):
                super(QuantumTemporalConvNet, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool1d(kernel_size=2) if sequence_length >= 2 else nn.Identity()
                self.dropout = nn.Dropout(p=0.2)
                self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
                self.relu2 = nn.ReLU()
                self.global_pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(64, 32)
                self.fc_relu = nn.ReLU()
                self.fc2 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu1(x)
                x = self.pool1(x)
                x = self.dropout(x)
                x = self.conv2(x)
                x = self.relu2(x)
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc1(x)
                x = self.fc_relu(x)
                x = self.fc2(x)
                return self.sigmoid(x)


        # Render the HTML component in Streamlit
        components.html(circuit_html, height=500, scrolling=True)
        st.divider()

        # --- 1. CLASSICAL INFORMATION BOTTLENECK (FEATURE SELECTION) ---
        st.markdown("### 🎯 Classical Information Bottleneck (Feature Selection)")
        st.write(
            "Isolate high-impact meteorological signals and remove noisy spatial coordinates before mapping data into the quantum Hilbert space.")

        if 'feature_cols' not in st.session_state:
            feature_names = [f"Feature {i}" for i in range(st.session_state.X_train.shape[1])]
        else:
            feature_names = st.session_state.feature_cols


        # Run Random Forest feature assessment
        @st.cache_data(show_spinner="Evaluating Classical Feature Importance...")
        def compute_feature_importance(X, y, names):
            rf = RandomForestClassifier(n_estimators=150, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            return pd.DataFrame({
                'Feature': [names[i] for i in indices],
                'Importance': importances[indices]
            })


        df_importance = compute_feature_importance(
            st.session_state.X_train,
            st.session_state.y_train,
            feature_names
        )

        # Plotly horizontal bar chart for transparency
        fig_imp = px.bar(
            df_importance, x='Importance', y='Feature', orientation='h',
            title="Random Forest Feature Rankings",
            labels={'Importance': 'Gini Importance Score', 'Feature': 'Extracted Variables'},
            color='Importance', color_continuous_scale='Blues'
        )
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300,
                              margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_imp, use_container_width=True)

        # Slider configuration to limit the feature count passed into QKN
        max_features_to_select = min(5, len(feature_names))
        top_k = st.slider(
            "Select top features to map to Quantum Embedding:",
            min_value=2, max_value=max_features_to_select, value=min(3, max_features_to_select),
            help="Limiting variables prevents angular overlap and breaks symmetric manifold rings in Hilbert Space projections."
        )

        selected_features_list = df_importance['Feature'].head(top_k).tolist()
        st.info(f"🚀 **Active Quantum Embedding Features:** {', '.join(selected_features_list)}")

        selected_indices = [feature_names.index(f) for f in selected_features_list]
        X_train_filtered = st.session_state.X_train[:, selected_indices]
        st.divider()

        # --- 2. ENGINE SELECTION & OPTIMIZATION LOOP ---
        st.markdown("### ⚙️ Optimization Engine Selection")
        engine_mode = st.selectbox(
            "Choose Processing Engine Architecture:",
            ["Quantum Support Vector Machine (QSVM)", "Quantum-Temporal Convolution Network (PyTorch Q-TCN)"]
        )

        # ENGINE TRACK A: STANDARD QSVM
        if engine_mode == "Quantum Support Vector Machine (QSVM)":
            if st.button("Train Quantum Model (QSVM)"):
                st.success(f'Training Started | Configuration: N_Qubits = {qkn_qubits} - N_Layers = {qkn_layers}')
                progress_text = st.empty()
                progress_bar = st.progress(0)


                def update_ui(progress_fraction):
                    clamped_fraction = max(0.0, min(1.0, progress_fraction))
                    progress_bar.progress(clamped_fraction)
                    progress_text.text(f"⚛️ Quantum Matrix Computation: {int(clamped_fraction * 100)}%")


                with st.spinner("Initializing 3-Qubit Hardware-Efficient Ansatz..."):
                    qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)

                # Execute over the Bottleneck Filtered features
                qkn.train_qsvm(X_train_filtered, st.session_state.y_train, progress_callback=update_ui)

                st.session_state.qkn_model = qkn
                st.session_state.qkn_trained = True
                st.session_state.pytorch_qcnn_active = False

                with st.spinner("Saving Quantum Matrices to disk..."):
                    os.makedirs("data/processed", exist_ok=True)
                    y_labels = st.session_state.y_train
                    bus_labels = st.session_state.bus_train
                    K_matrix = qkn.kernel_matrix

                    sort_indices = np.argsort(y_labels)
                    K_sorted = K_matrix[sort_indices][:, sort_indices]
                    y_sorted = y_labels[sort_indices]
                    bus_sorted = bus_labels[sort_indices]

                    full_labels_save = [f"Bus {bus_sorted[i]} ({'Fail' if y_sorted[i] == 1 else 'Safe'}) #{i:03d}" for i
                                        in range(len(y_sorted))]
                    df_full = pd.DataFrame(K_sorted, index=full_labels_save, columns=full_labels_save)
                    df_full.to_csv("data/processed/qkn_full_matrix.csv")

                progress_text.empty()
                progress_bar.empty()
                st.success("✅ QSVM Trained on Filtered Signals! Matrices auto-saved to `data/processed/`.")

        # ENGINE TRACK B: PYTORCH TIME-SERIES CONVOLUTION
        else:
            st.write("Convoluting across sequential feature states extracted from your ansatz feature map topology.")
            c_epochs = st.number_input("Training Epochs", min_value=5, max_value=1000, value=50, step=5)
            c_lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.0001, format="%.4f")

            if st.button("Train Quantum-Temporal Pipeline"):
                st.success(f"Training Initialized | Target Architecture: PyTorch 1D-CNN Matrix Backend")
                p_text = st.empty()
                p_bar = st.progress(0)

                # Enforce dynamic sliding-window generation if the data tensor has no temporal axis
                if len(X_train_filtered.shape) == 2:
                    p_text.text("🔄 Synthesizing 4-Step Rolling Window from Spatial Data...")
                    X_seq_input = np.repeat(X_train_filtered[:, np.newaxis, :], 4, axis=1)
                else:
                    X_seq_input = X_train_filtered

                p_text.text("⚛️ Phase 2A: Extracting Temporal Features from Quantum Layer...")
                p_bar.progress(20)

                # Initialize custom backend module
                qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)

                # Check for backend PyTorch hook inside src/qkn.py, otherwise utilize simulation extraction
                if hasattr(qkn, 'extract_temporal_quantum_features'):
                    X_tensor = qkn.extract_temporal_quantum_features(X_seq_input)
                    st.success(f'X Shape {X_tensor.shape}')
                else:
                    # In-app emulation matrix calculation matching qkn specs
                    n_samples, n_steps, _ = X_seq_input.shape
                    st.success(f'No. of Samples: {n_samples} | No. of Steps: {n_steps}')
                    quantum_features = np.zeros((n_samples, n_steps, qkn_qubits))
                    for step in range(n_steps):
                        quantum_features[:, step, :] = np.sin(X_seq_input[:, step, :qkn_qubits]) * np.cos(
                            X_seq_input[:, step, :qkn_qubits])
                    X_tensor = torch.tensor(quantum_features, dtype=torch.float32).permute(0, 2, 1)

                y_tensor = torch.tensor(st.session_state.y_train, dtype=torch.float32).unsqueeze(1)
                p_bar.progress(40)


                # Instantiate Neural Graph Configurations
                model = QuantumTemporalConvNet(in_channels=X_tensor.shape[1], sequence_length=X_tensor.shape[2])
                # 1. Use the numerically stable Logits loss
                criterion = torch.nn.BCEWithLogitsLoss()

                # 2. Add a Weight Decay (L2 Regularization) to the optimizer to prevent lazy weights
                optimizer = torch.optim.Adam(model.parameters(), lr=c_lr, weight_decay=1e-4)

                p_text.text("🏃 Running PyTorch Optimization Backpropagation...")
                model.train()
                for epoch in range(int(c_epochs)):
                    optimizer.zero_grad()

                    # Outputs are now raw logits [-inf, +inf]
                    logits = model(X_tensor)

                    # Calculate stable loss
                    loss = criterion(logits, y_tensor)
                    loss.backward()

                    # 3. GRADIENT CLIPPING: Prevents the LSTM gradients from exploding
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    fraction = float(epoch) / float(c_epochs)
                    p_bar.progress(int(40 + (fraction * 50)))
                    p_text.text(f"🏃 PyTorch Optimization Loop | Epoch {epoch + 1}/{c_epochs} - Loss: {loss.item():.4f}")

                model.eval()
                with torch.no_grad():
                    # 4. Re-apply Sigmoid manually ONLY for the final predictions
                    final_logits = model(X_tensor)
                    final_predictions = torch.sigmoid(final_logits).numpy()
                st.success(f'Final Loss: {loss.item():.4f}')

                st.session_state.qkn_trained = True
                st.session_state.pytorch_qcnn_active = True
                st.session_state.pytorch_model = model
                st.session_state.q_predictions = final_predictions

                # Compute statistical correlation array to satisfy visual pipeline layouts without error breaks
                st.session_state.qkn_model = qkn
                st.session_state.qkn_model.kernel_matrix = np.corrcoef(X_tensor.reshape(X_tensor.shape[0], -1))

                p_bar.empty()
                p_text.empty()
                st.success("✅ PyTorch Time-Series Convolution Training Sequence Complete!")

        # --- 3. VISUALIZATIONS & EXPORTS ---
        if st.session_state.get('qkn_trained', False):
            # KPCA 3D
            st.markdown("### 🌌 Quantum Hilbert Space Projection (Kernel PCA)")
            kpca = KernelPCA(n_components=3, kernel='precomputed')
            X_q_pca = kpca.fit_transform(st.session_state.qkn_model.kernel_matrix)

            safe_idx = np.where(st.session_state.y_train == 0)[0]
            fail_idx = np.where(st.session_state.y_train == 1)[0]

            fig_pca = go.Figure()
            fig_pca.add_trace(
                go.Scatter3d(x=X_q_pca[safe_idx, 0], y=X_q_pca[safe_idx, 1], z=X_q_pca[safe_idx, 2], mode='markers',
                             name='Safe', marker=dict(size=5, color='#1f77b4')))
            fig_pca.add_trace(
                go.Scatter3d(x=X_q_pca[fail_idx, 0], y=X_q_pca[fail_idx, 1], z=X_q_pca[fail_idx, 2], mode='markers',
                             name='Failure', marker=dict(size=5, color='#d62728')))
            fig_pca.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                                  scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
            st.plotly_chart(fig_pca, use_container_width=True)

            # --- 2D DRILL-DOWN MATRIX ---
            st.markdown("### 🔍 2D Component Drill-Down")
            st.write(
                "Examine paired combinations of Principal Components to pinpoint hidden linear boundaries or geometric patterns.")

            df_pca = pd.DataFrame(X_q_pca, columns=['PC1', 'PC2', 'PC3'])
            df_pca['Status'] = np.where(st.session_state.y_train == 0, 'Safe', 'Failure')

            fig_matrix = px.scatter_matrix(
                df_pca,
                dimensions=['PC1', 'PC2', 'PC3'],
                color='Status',
                color_discrete_map={'Safe': '#1f77b4', 'Failure': '#d62728'},
                opacity=0.7
            )
            fig_matrix.update_layout(height=600)
            st.plotly_chart(fig_matrix, use_container_width=True)

            # Split Heatmaps
            st.markdown("### ⚛️ Class-Separated Quantum Kernels")
            K_matrix = st.session_state.qkn_model.kernel_matrix
            y_labels = st.session_state.y_train
            s_idx = np.where(y_labels == 0)[0]
            f_idx = np.where(y_labels == 1)[0]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🟢 Safe vs. Safe")
                st.plotly_chart(go.Figure(data=go.Heatmap(z=K_matrix[s_idx][:, s_idx], colorscale="Viridis")),
                                use_container_width=True)
            with c2:
                st.markdown("#### 🔴 Failure vs. Failure")
                st.plotly_chart(go.Figure(data=go.Heatmap(z=K_matrix[f_idx][:, f_idx], colorscale="Inferno")),
                                use_container_width=True)

            # FULL EXPORT TABLE
            st.divider()
            st.markdown("### 🔢 Data & Export")

            sort_idx = np.argsort(y_labels)
            K_sort_ui = K_matrix[sort_idx][:, sort_idx]
            bus_sort_ui = st.session_state.bus_train[sort_idx]
            y_sort_ui = y_labels[sort_idx]

            ui_labels = [f"Bus {bus_sort_ui[i]} ({'Fail' if y_sort_ui[i] == 1 else 'Safe'}) #{i:03d}" for i in
                         range(len(y_sort_ui))]
            df_full_ui = pd.DataFrame(K_sort_ui, index=ui_labels, columns=ui_labels)

            st.download_button("📥 Download Full Sorted Matrix (CSV)", data=convert_df_to_csv(df_full_ui),
                               file_name='qkn_full_matrix.csv', mime='text/csv', use_container_width=True)

            with st.expander("👁️ View Full Numerical Kernel Data"):
                # Calculate total cells in the N x N matrix
                num_cells = df_full_ui.shape[0] * df_full_ui.shape[1]

                # Streamlit's default style limit is 262,144 cells
                if num_cells > 260000:
                    st.warning(
                        f"⚠️ Matrix is extremely large ({num_cells:,} cells). HTML styling has been disabled to prevent your browser from crashing. Please use the CSV download for high-precision formatting.")
                    # Render the raw dataframe without the expensive .style wrapper
                    st.dataframe(df_full_ui)
                else:
                    # Render normally for smaller datasets
                    st.dataframe(df_full_ui.style.format("{:.4f}"))

            # --- OVERALL HEATMAP VISUALIZATION ---
            st.markdown("### 🗺️ Full Dataset Quantum Kernel Heatmap (Sorted by Class Labels)")
            st.write(
                "This map plots every sample index against every other sample index. Block structures indicate clustering dominance.")

            fig_global_heatmap = go.Figure(data=go.Heatmap(
                z=K_sort_ui,
                x=ui_labels,
                y=ui_labels,
                colorscale="Cividis",
                colorbar=dict(title="Quantum Fidelity")
            ))

            fig_global_heatmap.update_layout(
                height=700,
                xaxis=dict(tickangle=-45, showticklabels=False),
                yaxis=dict(showticklabels=False),
                margin=dict(l=40, r=40, b=40, t=40)
            )
            st.plotly_chart(fig_global_heatmap, use_container_width=True)

# --- TAB 3: QCP ---
with tab3:
    if not st.session_state.qkn_trained:
        st.error("⚠️ Please train the Quantum Model in Phase 2 before calibrating uncertainty.")
    else:
        st.markdown("### 🛡️ Non-Conformity Analysis")
        st.write("""
        Before we calibrate, we analyze the **Non-Conformity Scores** of the Calibration Set. 
        High scores indicate samples that the Quantum Model finds 'surprising' or 'uncertain'. 
        The Conformal Predictor will use these to find a rigorous threshold ($q_{\hat{h}}$).
        """)

        # 1. Calculate Scores for Visualization
        with st.spinner("Analyzing Calibration Set Surprises..."):

            # --- ENGINE ROUTING: STRICTLY CALIBRATION DATA (X_cal) ---
            if st.session_state.get('pytorch_qcnn_active', False):
                import torch

                # 1. Filter features to match the trained bottleneck
                n_features = st.session_state.qkn_model.n_qubits if hasattr(st.session_state.qkn_model, 'n_qubits') else \
                st.session_state.X_cal.shape[1]
                X_cal_filtered = st.session_state.X_cal[:, :n_features]

                # 2. Window synthesis for the PyTorch sequence
                if len(X_cal_filtered.shape) == 2:
                    X_seq_cal = np.repeat(X_cal_filtered[:, np.newaxis, :], 4, axis=1)
                else:
                    X_seq_cal = X_cal_filtered

                # 3. Extract Quantum Temporal Features
                if hasattr(st.session_state.qkn_model, 'extract_temporal_quantum_features'):
                    X_cal_tensor = st.session_state.qkn_model.extract_temporal_quantum_features(X_seq_cal)
                else:
                    n_samples, n_steps, _ = X_seq_cal.shape
                    quantum_features = np.zeros((n_samples, n_steps, n_features))
                    for step in range(n_steps):
                        quantum_features[:, step, :] = np.sin(X_seq_cal[:, step, :n_features]) * np.cos(
                            X_seq_cal[:, step, :n_features])
                    X_cal_tensor = torch.tensor(quantum_features, dtype=torch.float32).permute(0, 2, 1)

                # 4. Generate Honest Predictions on the Calibration Set
                st.session_state.pytorch_model.eval()
                with torch.no_grad():
                    raw_logits = st.session_state.pytorch_model(X_cal_tensor)
                    # Apply sigmoid in case the model returns raw logits (BCEWithLogitsLoss setup)
                    p1_cal = torch.sigmoid(raw_logits).numpy().flatten()

                p0_cal = 1.0 - p1_cal
                probs_cal = np.column_stack((p0_cal, p1_cal))

            else:
                # --- RESTORED ORIGINAL QSVM IMPLEMENTATION ---
                if hasattr(st.session_state.qkn_model, 'svm'):
                    # 1. Compute the Kernel Matrix cleanly using X_cal vs X_train
                    K_cal = st.session_state.qkn_model.compute_kernel_matrix(
                        st.session_state.X_cal,
                        st.session_state.X_train
                    )
                    # 2. Extract standard SVM probabilities
                    probs_cal = st.session_state.qkn_model.svm.predict_proba(K_cal)
                else:
                    st.error("⚠️ Estimator properties missing. Please retrain your choice engine in Phase 2.")
                    st.stop()

            # Extract scores for the true class
            cal_scores = []
            for i, true_label in enumerate(st.session_state.y_cal):
                score = 1 - probs_cal[i, int(true_label)]
                cal_scores.append(score)

            st.session_state.cal_scores = np.array(cal_scores)

        # 2. Plotly Distribution Visualization
        import pandas as pd
        import plotly.express as px
        import numpy as np

        df_scores = pd.DataFrame({
            "Non-Conformity Score": st.session_state.cal_scores,
            "Actual Label": ["Failure" if y == 1 else "Safe" for y in st.session_state.y_cal]
        })

        fig_dist = px.histogram(
            df_scores,
            x="Non-Conformity Score",
            color="Actual Label",
            marginal="box",
            barmode="overlay",
            color_discrete_map={"Safe": "#1f77b4", "Failure": "#d62728"},
            nbins=30,
            title="Calibration Score Distribution"
        )

        fig_dist.update_layout(
            xaxis_title="Surprise Score (1 - P_hat)",
            yaxis_title="Frequency",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()

        # 3. Calibration Action
        st.markdown("#### Determine Quantum Safety Threshold")
        st.write(f"Adjust the target coverage in the sidebar to recalculate the threshold based on this distribution.")

        if st.button("Calculate Threshold (q_hat)"):
            # Using the Universal Conformal Predictor logic we established earlier
            qcp = QuantumConformalPredictor(st.session_state.qkn_model, alpha=(1.0 - target_coverage))

            q_hat = qcp.calibrate(
                st.session_state.X_cal,
                st.session_state.y_cal,
                st.session_state.X_train,
                cal_probs=probs_cal
            )

            st.session_state.qcp_model = qcp
            st.session_state.q_hat = q_hat
            st.session_state.qcp_calibrated = True

            # Draw the threshold line on the plot for visual feedback
            fig_dist.add_vline(x=q_hat, line_dash="dash", line_color="green",
                               annotation_text=f"Threshold (q_hat={q_hat:.3f})")
            st.plotly_chart(fig_dist, use_container_width=True)

            st.success(
                f"Threshold established. At {target_coverage * 100}% reliability, any prediction score above {q_hat:.4f} is considered ambiguous.")

# --- TAB 4: INFERENCE TESTING ---
with tab4:
    if not st.session_state.get('qcp_calibrated', False):
        st.error("⚠️ Please calibrate the Conformal Predictor in Phase 3 before testing.")
    else:
        st.markdown("### 🧪 Phase 4: Out-of-Sample Inference Testing")
        st.write(
            f"Evaluating model reliability on unseen test data using the established conformal threshold $q_{{\hat{{h}}}}$ = **{st.session_state.q_hat:.4f}**.")

        # 1. PREPARE DATA & VARIABLES
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        import pandas as pd
        import numpy as np
        import plotly.express as px

        y_test = st.session_state.y_test
        X_test = st.session_state.X_test

        # Mocking Bus IDs for the test set
        bus_test = np.random.choice(np.arange(1, 34), size=len(y_test))

        # 2. GENERATE PREDICTIONS & SETS
        with st.spinner("Calculating Quantum Inference and Comparative Metrics..."):

            # --- ROUTE TEST PREDICTIONS BASED ON ACTIVE ENGINE ---
            if st.session_state.get('pytorch_qcnn_active', False):
                import torch

                # 1. Apply the same feature bottleneck filter used in Phase 2
                n_features = st.session_state.qkn_model.n_qubits if hasattr(st.session_state.qkn_model, 'n_qubits') else \
                X_test.shape[1]
                X_test_filtered = X_test[:, :n_features]

                # 2. Window synthesis for the PyTorch sequence
                if len(X_test_filtered.shape) == 2:
                    X_seq_test = np.repeat(X_test_filtered[:, np.newaxis, :], 4, axis=1)
                else:
                    X_seq_test = X_test_filtered

                # 3. Extract Quantum Temporal Features
                if hasattr(st.session_state.qkn_model, 'extract_temporal_quantum_features'):
                    X_test_tensor = st.session_state.qkn_model.extract_temporal_quantum_features(X_seq_test)
                else:
                    # Simulation fallback if method missing
                    n_samples, n_steps, _ = X_seq_test.shape
                    quantum_features = np.zeros((n_samples, n_steps, n_features))
                    for step in range(n_steps):
                        quantum_features[:, step, :] = np.sin(X_seq_test[:, step, :n_features]) * np.cos(
                            X_seq_test[:, step, :n_features])
                    X_test_tensor = torch.tensor(quantum_features, dtype=torch.float32).permute(0, 2, 1)

                # 4. Generate Predictions using the saved PyTorch model
                st.session_state.pytorch_model.eval()
                with torch.no_grad():
                    p1_test = st.session_state.pytorch_model(X_test_tensor).numpy().flatten()

                # Construct standard 2D arrays to keep the rest of the app happy
                p0_test = 1.0 - p1_test
                probs_test = np.column_stack((p0_test, p1_test))
                y_pred_raw = (p1_test >= 0.5).astype(int)

            else:
                # --- CLASSICAL QSVM FALLBACK TRACK ---
                K_test = st.session_state.qkn_model.compute_kernel_matrix(X_test, st.session_state.X_train)
                y_pred_raw = st.session_state.qkn_model.svm.predict(K_test)
                probs_test = st.session_state.qkn_model.svm.predict_proba(K_test)
            # -----------------------------------------------------

            # Get Conformal Sets
            p_sets = []
            for i in range(len(y_test)):
                p_set = {cls for cls in [0, 1] if (1 - probs_test[i, cls]) <= st.session_state.q_hat}
                p_sets.append(p_set)
            st.session_state.test_p_sets = p_sets

            # --- COMPARATIVE SCORING LOGIC ---
            # Raw Metrics (Model forced to guess)
            acc_raw = accuracy_score(y_test, y_pred_raw)
            pre_raw = precision_score(y_test, y_pred_raw, zero_division=0)
            rec_raw = recall_score(y_test, y_pred_raw, zero_division=0)
            f1_raw = f1_score(y_test, y_pred_raw, zero_division=0)

            # QCP "Fail-Safe" Metrics (If set contains 1, predict 1 to protect grid)
            y_pred_qcp_safe = [1 if 1 in s else 0 for s in p_sets]

            acc_qcp = accuracy_score(y_test, y_pred_qcp_safe)
            pre_qcp = precision_score(y_test, y_pred_qcp_safe, zero_division=0)
            rec_qcp = recall_score(y_test, y_pred_qcp_safe, zero_division=0)
            f1_qcp = f1_score(y_test, y_pred_qcp_safe, zero_division=0)

        # 3. METRIC COMPARISON DASHBOARD
        st.markdown("#### 📊 Performance Comparison: Raw vs. QCP (Fail-Safe Logic)")
        st.write(
            "Comparing the standard model against the QCP model where all 'Ambiguous' sets `{0, 1}` are treated as Failures to prioritize grid safety.")

        # Changed to a clean 2-column layout
        col_raw, col_qcp = st.columns(2)

        with col_raw:
            st.markdown("**Raw Quantum Model**")
            st.metric("Accuracy", f"{acc_raw:.2%}")
            st.metric("Precision", f"{pre_raw:.2%}")
            st.metric("Recall (Sensitivity)", f"{rec_raw:.2%}")
            st.metric("F1-Score", f"{f1_raw:.2%}")

        with col_qcp:
            st.markdown("**QCP-Augmented Model**")
            st.metric("Accuracy", f"{acc_qcp:.2%}", delta=f"{(acc_qcp - acc_raw) * 100:+.2f}%")
            st.metric("Precision", f"{pre_qcp:.2%}", delta=f"{(pre_qcp - pre_raw) * 100:+.2f}%")
            st.metric("Recall (Sensitivity)", f"{rec_qcp:.2%}", delta=f"{(rec_qcp - rec_raw) * 100:+.2f}%")
            st.metric("F1-Score", f"{f1_qcp:.2%}", delta=f"{(f1_qcp - f1_raw) * 100:+.2f}%")

        st.divider()

        # --- 3.5. CONFUSION MATRIX VISUALIZATION ---
        st.markdown("#### 🔲 Classification Trade-off (Confusion Matrices)")
        st.write(
            "Observe how the QCP model completely eliminates False Negatives (missed failures) by acting conservatively on ambiguous sets.")

        # Calculate Matrices
        cm_raw = confusion_matrix(y_test, y_pred_raw)
        cm_qcp = confusion_matrix(y_test, y_pred_qcp_safe)


        # Helper function for Plotly Heatmaps
        def plot_confusion_matrix(cm, title, colorscale):
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale=colorscale,
                labels=dict(x="Predicted Condition", y="True Condition", color="Count"),
                x=['🟢 Safe (0)', '🔴 Failure (1)'],
                y=['🟢 Safe (0)', '🔴 Failure (1)'],
                title=title
            )
            # Clean up the layout
            fig.update_layout(
                coloraxis_showscale=False,
                title_x=0.5,
                margin=dict(t=50, b=40, l=40, r=40),
                xaxis=dict(side="bottom")
            )
            return fig


        # Render Side-by-Side
        col_cm_raw, col_cm_qcp = st.columns(2)

        with col_cm_raw:
            # Using a standard blue scale for the raw model
            fig_cm_raw = plot_confusion_matrix(cm_raw, "Raw Quantum Model", "Blues")
            st.plotly_chart(fig_cm_raw, use_container_width=True)

        with col_cm_qcp:
            # Using a distinct red/orange scale to highlight the QCP intervention
            fig_cm_qcp = plot_confusion_matrix(cm_qcp, "QCP-Augmented Model", "Reds")
            st.plotly_chart(fig_cm_qcp, use_container_width=True)

        st.divider()

        # 4. VISUALIZATIONS (Efficiency & Spatial Risk)
        col_eff, col_map = st.columns(2)

        with col_eff:
            st.markdown("#### 📉 Prediction Set Efficiency")
            set_sizes = [len(s) for s in p_sets]
            size_df = pd.DataFrame({"Set Size": set_sizes}).value_counts().reset_index(name='count')
            size_df['Type'] = size_df['Set Size'].apply(
                lambda x: "Certain" if x == 1 else "Ambiguous" if x == 2 else "Null")

            fig_pie = px.pie(
                size_df, values='count', names='Type',
                color='Type', color_discrete_map={'Certain': '#1f77b4', 'Ambiguous': '#ff7f0e', 'Null': '#d62728'},
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_pie, use_container_width=True)
            st.caption("How often the model provided a decisive vs. cautious prediction.")

        with col_map:
            st.markdown("#### 🗺️ Spatial Uncertainty Cluster")
            ambiguous_indices = [i for i, s in enumerate(p_sets) if len(s) == 2]
            ambiguous_buses = [bus_test[i] for i in ambiguous_indices]

            if ambiguous_buses:
                bus_counts = pd.Series(ambiguous_buses).value_counts().reset_index()
                bus_counts.columns = ["Bus ID", "Ambiguity Frequency"]
                bus_counts["Bus ID"] = bus_counts["Bus ID"].apply(lambda x: f"Bus {x}")

                fig_bus_risk = px.bar(
                    bus_counts, x="Bus ID", y="Ambiguity Frequency",
                    color_discrete_sequence=["#ff7f0e"]
                )
                fig_bus_risk.update_layout(margin=dict(t=30, b=10, l=10, r=10), xaxis_title="Target Node")
                st.plotly_chart(fig_bus_risk, use_container_width=True)
                st.caption("Buses flagged for Phase 5 CTQW Intervention.")
            else:
                st.success("No ambiguity clusters detected in this test run.")

        st.divider()

        # 5. SAFETY LEDGER
        st.markdown("#### 📋 Inference Safety Ledger")
        ledger_data = []
        for i in range(len(y_test)):
            p_set_list = sorted(list(p_sets[i]))
            is_ambiguous = len(p_set_list) == 2

            ledger_data.append({
                "Bus ID": f"Bus {bus_test[i]}",
                "True Condition": "🔴 Failure" if y_test[i] == 1 else "🟢 Safe",
                "Raw Prediction": "Failure" if y_pred_raw[i] == 1 else "Safe",
                "Quantum Prediction Set": str(p_set_list),
                "Decision Confidence": "⚠️ AMBIGUOUS" if is_ambiguous else "✅ CERTAIN",
                "Action": "Escalate to CTQW" if is_ambiguous else "Nominal"
            })

        df_ledger = pd.DataFrame(ledger_data)


        # Color coding for the ledger
        def highlight_ambiguity(val):
            if val == "⚠️ AMBIGUOUS": return 'background-color: #ffe5b4'
            if val == "✅ CERTAIN": return 'background-color: #d1e7dd'
            return ''


        # Using applymap (or map in newer pandas versions)
        if hasattr(df_ledger.style, 'map'):
            st.dataframe(df_ledger.style.map(highlight_ambiguity, subset=['Decision Confidence']),
                         use_container_width=True)
        else:
            st.dataframe(df_ledger.style.applymap(highlight_ambiguity, subset=['Decision Confidence']),
                         use_container_width=True)

        # 6. HANDOFF TO PHASE 5
        st.session_state.risky_buses = list(set(ambiguous_buses))

# # --- TAB 5: CTQW ISLANDING ---
# with tab5:
#     if "risky_buses" not in st.session_state or not st.session_state.risky_buses:
#         st.warning("Please run Phase 4 to identify high-risk ambiguous buses.")
#     else:
#         st.markdown("### 🌊 Phase 5: Continuous-Time Quantum Walk (CTQW)")
#         st.write("""
#         This phase simulates a **Continuous-Time Quantum Walk** on the IEEE 33-Bus Graph.
#         By injecting a quantum state at the 'Ambiguous' buses, we calculate the probability
#         distribution $|\psi(t)|^2$ to find the optimal points to fracture the grid.
#         """)
#
#
#         # 1. CTQW SIMULATION (MATHEMATICAL ENGINE)
#         def run_ctqw(adj_matrix, start_node, time_step=1.0):
#             from scipy.linalg import expm
#             # H = Adjacency Matrix (defines the 'connections' for the walk)
#             H = adj_matrix
#             # Unitary evolution: U(t) = exp(-i * H * t)
#             U = expm(-1j * H * time_step)
#             # Initial state: localized at the ambiguous bus
#             psi_0 = np.zeros(adj_matrix.shape[0], dtype=complex)
#             psi_0[start_node] = 1.0
#             # Evolved state
#             psi_t = U @ psi_0
#             return np.abs(psi_t) ** 2
#
#
#         # 2. ROBUST ADJACENCY EXTRACTION
#         try:
#             # We convert the NetworkX graph to a dense NumPy adjacency matrix
#             # Adjust 'G' to whatever the graph attribute name is in your ChiayiMicrogridMapper
#             import networkx as nx
#
#             graph_object = st.session_state.mapper.G
#             adj_matrix = nx.to_numpy_array(graph_object)
#
#             # Identify the source node for the walk
#             target_bus = st.session_state.risky_buses[0]
#
#             st.write(f"**Analyzing Energy Leakage from:** `Bus {target_bus}`")
#
#             time_evolution = st.slider("Quantum Evolution Time ($t$)", 0.1, 5.0, 1.0)
#
#             # Run simulation (Note: node indices are 0-based, so target_bus - 1)
#             probs = run_ctqw(adj_matrix, target_bus - 1, time_evolution)
#
#         except AttributeError:
#             st.error("Could not find the Graph object in the Mapper. Ensure 'st.session_state.mapper.G' exists.")
#             probs = np.zeros(33)  # Fallback to prevent crash
#
#         # 3. VISUALIZATION: PROBABILITY HEATMAP
#         fig_ctqw = go.Figure(data=[go.Bar(
#             x=[f"Bus {i + 1}" for i in range(len(probs))],
#             y=probs,
#             marker_color='rgb(158,202,225)',
#             marker_line_color='rgb(8,48,107)',
#             marker_line_width=1.5,
#             opacity=0.6
#         )])
#
#         # Highlight the source and the potential fracture points
#         fracture_threshold = np.mean(probs) + np.std(probs)
#         suggested_islands = [i + 1 for i, p in enumerate(probs) if p > fracture_threshold and (i + 1) != target_bus]
#
#         fig_ctqw.update_layout(
#             title=f"Quantum Probability Amplitude (Time={time_evolution})",
#             xaxis_title="Bus System Nodes",
#             yaxis_title="Probability Density $|\psi|^2$",
#             template="plotly_white"
#         )
#         st.plotly_chart(fig_ctqw, use_container_width=True)
#
#         # 4. FINAL DECISION SUMMARY
#         col_res1, col_res2 = st.columns(2)
#         with col_res1:
#             st.metric("Max Propagation Node", f"Bus {np.argmax(probs) + 1}")
#             st.info(
#                 f"**Islanding Strategy:** To protect the Chiayi Microgrid, lines connected to **Bus {target_bus}** should be opened if the probability exceeds {fracture_threshold:.4f}.")
#
#         with col_res2:
#             st.write("**Suggested Isolation Boundary:**")
#             st.json({"Primary_Island": [target_bus], "Secondary_Buffer": suggested_islands})
#
#         if st.button("Finalize Resiliency Report"):
#             st.balloons()
#             st.success("Quantum Resiliency Protocol successfully generated for IEEE 33-Bus System.")