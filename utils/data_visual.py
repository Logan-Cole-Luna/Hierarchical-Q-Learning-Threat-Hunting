import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Define the directory containing your CSV files
data_directory = 'data/'

# Sampling fraction (e.g., divide by 5)
sampling_fraction = 0.2

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
        
        # Sample within each 'Label' category to retain proportional representation
        temp_data = temp_data.groupby('Label', group_keys=False).apply(
            lambda x: x.sample(frac=sampling_fraction, random_state=42)
        ).reset_index(drop=True)
        
        all_data = pd.concat([all_data, temp_data], ignore_index=True)

# Separate data with and without "BENIGN" label
data_with_benign = all_data
data_without_benign = all_data[all_data['Label'] != 'BENIGN']

# Define a color map for each attack type
attack_types = all_data['Attack_Type'].unique()
colors = ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for _ in range(6)]) for _ in attack_types]
color_map = dict(zip(attack_types, colors))

# Define a function to create a Sankey diagram
def create_sankey(data, title):
    # Group data by Destination Port, Attack_Type, and Label, aggregating necessary metrics
    sankey_data = data.groupby(['Destination Port', 'Attack_Type', 'Label']).agg({
        'Flow Duration': 'sum',
        'Total Fwd Packets': 'sum',
        'Total Backward Packets': 'sum',
        'Total Length of Fwd Packets': 'sum',
        'Total Length of Bwd Packets': 'sum'
    }).reset_index()

    # Define nodes and links for the Sankey diagram
    nodes = list(sankey_data['Destination Port'].astype(str).unique()) + \
            list(sankey_data['Attack_Type'].unique()) + \
            list(sankey_data['Label'].unique())

    # Create node indices for Sankey plot
    node_indices = {node: idx for idx, node in enumerate(nodes)}

    # Define source, target, and values for each link
    source = sankey_data['Destination Port'].astype(str).map(node_indices).tolist()
    target_attack = sankey_data['Attack_Type'].map(node_indices).tolist()
    target_label = sankey_data['Label'].map(node_indices).tolist()
    values = sankey_data['Flow Duration'].tolist()

    # Combining source-target lists for both Attack_Type and Label connections
    source_combined = source + target_attack
    target_combined = target_attack + target_label
    values_combined = values + values  # Duplicate values for two connections

    # Apply colors based on attack type for nodes
    node_colors = []
    for node in nodes:
        if node in sankey_data['Attack_Type'].unique():
            node_colors.append(color_map[node])  # Color based on attack type
        elif "Port" in str(node):
            node_colors.append("#1f77b4")  # Blue for destination ports
        else:
            node_colors.append("#2ca02c")  # Green for labels (e.g., BENIGN)

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,  # Increase padding for more space between nodes
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors
        ),
        link=dict(
            source=source_combined,
            target=target_combined,
            value=values_combined,
            color="rgba(100, 100, 100, 0.5)",
        )
    )])

    # Update layout for readability
    fig.update_layout(
        title_text=title,
        font_size=10,
        plot_bgcolor="white",
        width=1200,
        height=800
    )

    # Display the figure
    fig.show()

# Generate two visualizations: one with and one without "BENIGN"
create_sankey(data_with_benign, "Network Traffic Flow Visualization Including BENIGN")
create_sankey(data_without_benign, "Network Traffic Flow Visualization Excluding BENIGN")
