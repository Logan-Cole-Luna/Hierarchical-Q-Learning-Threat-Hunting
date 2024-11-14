import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Define the directory containing your CSV files
data_directory = 'data/'  # Replace this with your actual directory path

# Sampling fraction for non-DDoS attack types
sampling_fraction = 0.05

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Loop through each file in the directory and read CSV files
for filename in os.listdir(data_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(data_directory, filename)
        # Load each CSV file and add a column for 'Attack_Type' based on filename
        temp_data = pd.read_csv(file_path)
        temp_data.columns = temp_data.columns.str.strip()  # Clean column names
        temp_data['Attack_Type'] = filename.split('-')[1]  # Extract attack type from filename
        
        all_data = pd.concat([all_data, temp_data], ignore_index=True)

# Separate DDoS data and non-DDoS data
ddos_data = all_data[all_data['Label'].str.contains('DDoS')]
non_ddos_data = all_data[~all_data['Label'].str.contains('DDoS')]

# Create datasets for each visualization
# 1. Full DDoS and BENIGN included, non-DDoS downsampled
data_with_benign = pd.concat([ddos_data, non_ddos_data.groupby('Label', group_keys=False).apply(
    lambda x: x.sample(frac=sampling_fraction, random_state=42))])

# 2. Full DDoS without BENIGN, non-DDoS downsampled
data_without_benign = data_with_benign[data_with_benign['Label'] != 'BENIGN']

# 3. Downsample everything (including DDoS) and exclude BENIGN
data_downsampled = all_data[all_data['Label'] != 'BENIGN'].groupby('Label', group_keys=False).apply(
    lambda x: x.sample(frac=sampling_fraction, random_state=42))

# Define a color map for each attack type
attack_types = all_data['Attack_Type'].unique()
colors = ["#" + ''.join(np.random.choice(list("0123456789ABCDEF"), 6)) for _ in attack_types]
color_map = dict(zip(attack_types, colors))

# Define a function to create a Sankey diagram
def create_sankey(data, title):
    sankey_data = data.groupby(['Destination Port', 'Attack_Type', 'Label']).agg({
        'Flow Duration': 'sum',
        'Total Fwd Packets': 'sum',
        'Total Backward Packets': 'sum',
        'Total Length of Fwd Packets': 'sum',
        'Total Length of Bwd Packets': 'sum'
    }).reset_index()

    nodes = list(sankey_data['Destination Port'].astype(str).unique()) + \
            list(sankey_data['Attack_Type'].unique()) + \
            list(sankey_data['Label'].unique())
    node_indices = {node: idx for idx, node in enumerate(nodes)}

    source = sankey_data['Destination Port'].astype(str).map(node_indices).tolist()
    target_attack = sankey_data['Attack_Type'].map(node_indices).tolist()
    target_label = sankey_data['Label'].map(node_indices).tolist()
    values = sankey_data['Flow Duration'].tolist()
    values_log_scaled = [np.log1p(value) for value in values]
    
    source_combined = source + target_attack
    target_combined = target_attack + target_label
    values_combined = values_log_scaled + values_log_scaled

    node_colors = [color_map.get(node, "#2ca02c") if node in color_map else "#1f77b4" for node in nodes]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=50,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source_combined,
            target=target_combined,
            value=values_combined,
            color="rgba(100, 100, 100, 0.3)",
        )
    )])

    fig.update_layout(
        title_text=title,
        font_size=10,
        plot_bgcolor="white",
        width=1200,
        height=1000
    )
    fig.show()

# Generate three Sankey diagrams
create_sankey(data_with_benign, "Network Traffic Flow Visualization Including BENIGN with Full DDoS")
create_sankey(data_without_benign, "Network Traffic Flow Visualization Excluding BENIGN with Full DDoS")
create_sankey(data_downsampled, "Network Traffic Flow Visualization Excluding BENIGN with Downsampled DDoS")
