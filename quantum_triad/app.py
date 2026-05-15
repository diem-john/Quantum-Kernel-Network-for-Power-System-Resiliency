import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import pennylane as qml
import warnings

warnings.filterwarnings('ignore')

# Custom Modules
from src.mapping import ChiayiMicrogridMapper, QuantumWalkIslandingMapper
from src.qkn import QuantumKernelNetwork
from src.qcp import QuantumConformalPredictor


# --- HELPER: CSV EXPORT ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a Pandas DataFrame to a UTF-8 encoded CSV."""
    return df.to_csv(index=True).encode('utf-8')


# --- PAGE CONFIG ---
st.set_page_config(page_title="Q-Rating Chaos", layout="wide")
st.title("⚡ Q-Rating Chaos: A Tri-Partite Quantum Framework for Typhoon Modeling and Microgrid Resilience")
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

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Pipeline Controls")
n_samples = st.sidebar.slider("Sample Size (POC Speed)", 50, 500, 150)
qkn_qubits = st.sidebar.selectbox("QKN Qubits", [3, 4, 5], index=0)
qkn_layers = st.sidebar.slider("QKN Entangling Layers", 1, 5, 2)
target_coverage = st.sidebar.slider("QCP Target Coverage (%)", 80, 99, 90) / 100.0

# --- TABS LAYOUT ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📍 Phase 1: Mapping",
    "🧠 Phase 2: QKN Training",
    "🛡️ Phase 3: QCP Calibration",
    "🌊 Phase 4: CTQW Islanding"
])

# --- TAB 1: MAPPING ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_interactive_map(st.session_state.mapper, "Enhanced IEEE 33-Bus System in Chiayi"),
                        use_container_width=True)
    with col2:
        st.info(
            "**Geospatial Setup**\n\nThe 33-bus system is mapped to Chiayi. Coastal exposure increases vulnerability for buses further West. The map is fully interactive—you can pan and zoom across the real streets.")

        if st.button("Load & Extract Typhoon Data"):
            with st.spinner("Processing Chronological Data..."):
                # MOCK DATA GENERATOR FOR STREAMLIT SPEED
                X = np.random.uniform(-np.pi, np.pi, (n_samples, 3))
                Y = np.random.randint(0, 2, n_samples)
                train_b, cal_b = int(0.6 * n_samples), int(0.8 * n_samples)

                st.session_state.X_train, st.session_state.y_train = X[:train_b], Y[:train_b]
                st.session_state.X_cal, st.session_state.y_cal = X[train_b:cal_b], Y[train_b:cal_b]
                st.session_state.X_test, st.session_state.y_test = X[cal_b:], Y[cal_b:]
                st.session_state.data_loaded = True
                st.success("Data Splitting Complete (Train/Cal/Test). Proceed to Phase 2.")

# --- TAB 2: QKN ---
with tab2:
    if not st.session_state.data_loaded:
        st.error("⚠️ Please Load & Extract Typhoon Data in Phase 1 before proceeding.")
    else:
        st.write(
            "Using the extracted meteorological and spatial features to train the Quantum Support Vector Machine via Fidelity Estimation.")

        # --- DYNAMIC D3.JS CIRCUIT VISUALIZATION ---
        st.markdown("### 🧬 Quantum Circuit Architecture (Ansatz)")
        st.write(
            "Interact with the sliders to see how the entanglement topology scales for different hardware configurations.")

        # The HTML/JS code for the D3.js Interactive Circuit
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

        # Render the HTML component in Streamlit
        components.html(circuit_html, height=500, scrolling=True)
        st.divider()

        # --- TRAINING BUTTON ---
        if st.button("Train Quantum Model (QSVM)"):
            # 1. Create empty placeholders for the UI
            progress_text = st.empty()
            progress_bar = st.progress(0)

            # 2. Define the callback function that Streamlit will run
            def update_ui(progress_fraction):
                # Ensure the value stays safely between 0.0 and 1.0
                clamped_fraction = max(0.0, min(1.0, progress_fraction))
                progress_bar.progress(clamped_fraction)
                progress_text.text(f"Executing Quantum Circuits... {int(clamped_fraction * 100)}%")


            # 3. Execute the training
            with st.spinner("Initializing Hilbert Space..."):
                # Use the actual Streamlit sidebar values for training
                qkn = QuantumKernelNetwork(n_qubits=qkn_qubits, layers=qkn_layers)

                # Pass the update_ui function into the trainer
                qkn.train_qsvm(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    progress_callback=update_ui
                )

                st.session_state.qkn_model = qkn
                st.session_state.qkn_trained = True

            # 4. Clean up the UI on completion
            progress_text.empty()
            progress_bar.empty()
            st.success("QSVM Trained using Fidelity Quantum Kernel! Proceed to Phase 3.")

            # --- INTERACTIVE QUANTUM KERNEL VISUALIZATION ---
            if st.session_state.qkn_trained:
                st.divider()
                st.markdown("### ⚛️ Class-Separated Quantum Kernel Matrices")
                st.write("""
                    By separating the Gram matrix, we can evaluate the **Intra-Class Quantum Fidelity**. 
                    Ideally, both matrices should display high values (bright colors), proving that the quantum feature map successfully clusters similar weather-grid states together in the Hilbert space.
                    """)

                # Extract the raw matrix and labels
                K_matrix = st.session_state.qkn_model.kernel_matrix
                y_labels = st.session_state.y_train

                # 1. Isolate the indices for each class
                safe_indices = np.where(y_labels == 0)[0]
                fail_indices = np.where(y_labels == 1)[0]

                # 2. Extract the sub-matrices using NumPy slicing
                K_safe = K_matrix[safe_indices][:, safe_indices]
                K_fail = K_matrix[fail_indices][:, fail_indices]

                # 3. Create a side-by-side layout in Streamlit
                col_safe, col_fail = st.columns(2)

                with col_safe:
                    st.markdown("#### 🟢 Safe vs. Safe Microgrids")
                    # Generate hover text
                    hover_safe = [[f"Row ID: {r}<br>Col ID: {c}<br>Fidelity: {K_safe[r, c]:.4f}"
                                   for c in range(len(safe_indices))] for r in range(len(safe_indices))]

                    # Use 'Viridis' (Green/Blue/Yellow) for Safe
                    fig_safe = go.Figure(data=go.Heatmap(
                        z=K_safe, text=hover_safe, hoverinfo="text", colorscale="Viridis"
                    ))
                    fig_safe.update_layout(
                        width=400, height=400, margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Safe Instance ID", yaxis_title="Safe Instance ID"
                    )
                    st.plotly_chart(fig_safe, use_container_width=True)

                with col_fail:
                    st.markdown("#### 🔴 Failure vs. Failure Microgrids")
                    # Generate hover text
                    hover_fail = [[f"Row ID: {r}<br>Col ID: {c}<br>Fidelity: {K_fail[r, c]:.4f}"
                                   for c in range(len(fail_indices))] for r in range(len(fail_indices))]

                    # Use 'Inferno' (Black/Red/Yellow) for Failure
                    fig_fail = go.Figure(data=go.Heatmap(
                        z=K_fail, text=hover_fail, hoverinfo="text", colorscale="Inferno"
                    ))
                    fig_fail.update_layout(
                        width=400, height=400, margin=dict(l=20, r=20, t=20, b=20),
                        xaxis_title="Failure Instance ID", yaxis_title="Failure Instance ID"
                    )
                    st.plotly_chart(fig_fail, use_container_width=True)

                # --- 2. RAW DATA TABLES & EXPORT ---
                st.divider()
                st.markdown("#### 🔢 Raw Quantum Kernel Output Data & Export")
                st.write("Inspect the raw floating-point fidelity values or download the matrices as CSV files for external analysis.")

                # Format the DataFrames
                df_safe = pd.DataFrame(K_safe)
                df_safe.index = [f"Safe_{i}" for i in range(len(safe_indices))]
                df_safe.columns = [f"Safe_{i}" for i in range(len(safe_indices))]

                df_fail = pd.DataFrame(K_fail)
                df_fail.index = [f"Fail_{i}" for i in range(len(fail_indices))]
                df_fail.columns = [f"Fail_{i}" for i in range(len(fail_indices))]

                # Create CSV data using the cached helper function
                csv_safe = convert_df_to_csv(df_safe)
                csv_fail = convert_df_to_csv(df_fail)

                # Layout the columns for viewing and downloading
                col_table_safe, col_table_fail = st.columns(2)

                with col_table_safe:
                    # Add the download button
                    st.download_button(
                        label="📥 Download Safe Matrix (CSV)",
                        data=csv_safe,
                        file_name='qkn_safe_matrix.csv',
                        mime='text/csv',
                        use_container_width=True # Makes the button stretch nicely
                    )
                    with st.expander("👁️ View Safe Matrix Raw Data"):
                        st.dataframe(df_safe.style.format("{:.4f}"))

                with col_table_fail:
                    # Add the download button
                    st.download_button(
                        label="📥 Download Failure Matrix (CSV)",
                        data=csv_fail,
                        file_name='qkn_failure_matrix.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                    with st.expander("👁️ View Failure Matrix Raw Data"):
                        st.dataframe(df_fail.style.format("{:.4f}"))

# --- TAB 3: QCP ---
with tab3:
    if not st.session_state.qkn_trained:
        st.error("⚠️ Please train the Quantum Model in Phase 2 before calibrating uncertainty.")
    else:
        st.write(
            "Applying Split Conformal Prediction to bound the quantum model's uncertainty with a mathematically rigorous guarantee.")
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

            coverage = sum(
                1 for i, p_set in enumerate(st.session_state.pred_sets) if st.session_state.y_test[i] in p_set)
            ambiguous = sum(1 for p_set in st.session_state.pred_sets if len(p_set) == 2)

            c1, c2, c3 = st.columns(3)
            c1.metric("Target Coverage", f"{target_coverage * 100:.1f}%")
            c2.metric("Empirical Coverage", f"{(coverage / len(st.session_state.y_test)) * 100:.1f}%")
            c3.metric("Ambiguous Alerts (Set=2)", f"{ambiguous}")
            st.success("Calibration complete. Ambiguous alerts identified. Proceed to Phase 4.")

# --- TAB 4: CTQW ---
with tab4:
    if not st.session_state.qcp_calibrated:
        st.error("⚠️ Please calibrate the Conformal Predictor in Phase 3 to generate failure predictions.")
    else:
        st.write(
            "Using the ambiguous alerts from the QCP, we simulate line fractures and use a Continuous-Time Quantum Walk to identify survivable microgrid islands.")

        default_breaks = ["(2, 19)", "(6, 26)"]
        all_edges = [str(e) for e in st.session_state.mapper.graph.edges()]
        selected_breaks = st.multiselect("Select lines predicted to fail by QKN:", all_edges, default=default_breaks)

        if st.button("Simulate Quantum Walk Islanding"):
            with st.spinner("Evolving Quantum Hamiltonian..."):
                failed_edges = [eval(e) for e in selected_breaks]
                post_grid = st.session_state.mapper.simulate_typhoon_failures(failed_edges)

                qw_mapper = QuantumWalkIslandingMapper(post_grid)
                zones = qw_mapper.identify_islands()

                # Assign colors to different zones for visualization
                colors = px.colors.qualitative.Safe  # Colorblind-friendly palette
                node_colors = {}
                zone_metrics = []

                for i, (zone_id, data) in enumerate(zones.items()):
                    color = colors[i % len(colors)]
                    for node in data['nodes']:
                        node_colors[node] = color
                    zone_metrics.append({"Zone": zone_id, "Size (Nodes)": data['size'], "Buses": str(data['nodes'])})

                col_map, col_data = st.columns([2, 1])
                with col_map:
                    st.plotly_chart(plot_interactive_map(st.session_state.mapper, "Fractured Microgrid Zones",
                                                         node_colors=node_colors), use_container_width=True)
                with col_data:
                    st.dataframe(pd.DataFrame(zone_metrics), hide_index=True)
                    st.warning(
                        f"**Action Required:** Dispatch local DERs and battery storage to stabilize {len(zones) - 1} isolated islands.")