import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define the knowledge graph
vSTM_knowledge_graph = {
    "market_equilibrium": np.array([1, 0, 0, 0, 0]), 
    "price_increase": np.array([0, 1, 0, 0, 0]),
    "qd_decrease": np.array([0, 0, 1, 0, 0]), 
    "qs_increase": np.array([0, 0, 0, 1, 0]), 
    "surplus": np.array([0, 0, 0, 0, 1]),
    "market_disequilibrium": np.array([1, 1, 1, 1, 1])
}

# Initialize the graph
vstm = nx.DiGraph()

# Add nodes to the graph with their vector attributes
for node, vector in vSTM_knowledge_graph.items():
    vstm.add_node(node, vector=vector)
# Add nodes to the graph
vstm.add_nodes_from(vSTM_knowledge_graph)

# Define edges (for illustration purposes, you can define your own relationships)
edges = [
    ("market_equilibrium", "price_increase"),
    ("price_increase", "qd_decrease"),
    ("price_increase", "qs_increase"),
    ("qd_decrease", "surplus"),
    ("qs_increase", "surplus"),
    ("surplus", "market_disequilibrium")
]

# Add edges to the graph
vstm.add_edges_from(edges)

# Function to draw the graph
def draw_graph(G, node_labels=True, edge_labels=False, node_color='lightblue', edge_color='gray', node_size=500, font_size=8, label_offset=0.05):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color)
    
    if node_labels:
        labels = {node: f"{node}\n{G.nodes[node]['vector']}" for node in G.nodes()}
        label_pos = {node: (x + label_offset, y) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(G, label_pos, labels, font_size=font_size)
    
    if edge_labels:
        edge_labels = nx.get_edge_attributes(G, 'label')
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size)
    
    plt.show()

# Create a blank image of size 512x512
image = np.zeros((512, 512, 3), np.uint8)

# Helper function to draw a line (adjusted y-coordinates to match matplotlib's coordinate system)
def draw_line(image, start, end, color=(255, 255, 255), thickness=1):
    start_adjusted = (start[0], image.shape[0] - start[1])  # Adjust y-coordinate
    end_adjusted = (end[0], image.shape[0] - end[1])  # Adjust y-coordinate
    return cv2.line(image, start_adjusted, end_adjusted, color, thickness)

# Helper function to write text (adjusted y-coordinate)
def write_text(image, text, position, color=(255, 255, 255), thickness=1):
    position_adjusted = (position[0], image.shape[0] - position[1])  # Adjust y-coordinate
    return cv2.putText(image, text, position_adjusted, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

# Helper function to draw half-rectangle border (adjusted y-coordinates)
def draw_half_rectangle(image, top_left, bottom_right, color, thickness=1):
    top_left_adjusted = (top_left[0], image.shape[0] - top_left[1])  # Adjust y-coordinate
    bottom_right_adjusted = (bottom_right[0], image.shape[0] - bottom_right[1])  # Adjust y-coordinate
    pts = np.array([[top_left_adjusted[0], bottom_right_adjusted[1]],
                    [top_left_adjusted[0], top_left_adjusted[1]],
                    [bottom_right_adjusted[0], top_left_adjusted[1]],
                    [bottom_right_adjusted[0], bottom_right_adjusted[1]]], np.int32)
    return cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

# Define the pltm_dict with image operations as lambdas
pltm_dict = {
    tuple([1, 0, 0, 0, 0]): [  # tuple for market equilibrium
        lambda img: draw_line(img, (150, 400), (400, 150)),  # demand curve
        lambda img: draw_line(img, (150, 150), (400, 400)),  # supply curve
        lambda img: draw_line(img, (100, 450), (100, 100)),  # price axis
        lambda img: draw_line(img, (100, 100), (450, 100)),  # quantity axis
        lambda img: draw_line(img, (100, 275), (275, 275)),  # equilibrium price line
        lambda img: draw_line(img, (275, 275), (275, 100)),  # equilibrium quantity line
        lambda img: write_text(img, "P", (50, 450)),  # price axis label
        lambda img: write_text(img, "Q", (450, 50)),  # quantity axis label
        lambda img: write_text(img, "D", (125, 410)),  # demand curve label
        lambda img: write_text(img, "S", (410, 410)),  # supply curve label
        lambda img: write_text(img, "p1", (50, 265)),  # price 1 label
        lambda img: write_text(img, "q", (255, 60)),  # price 2 label
    ],
    tuple([0, 1, 0, 0, 0]): [  # tuple for price increase
        lambda img: draw_line(img, (100, 338), (338, 338)),  # price increase, qs line
        lambda img: draw_line(img, (100, 338), (213, 338)),  # price increase, qd line
        lambda img: write_text(img, "p2", (50, 330)),  # price axis label
    ],
    tuple([0, 0, 1, 0, 0]): [  # tuple for quantity demanded decrease
        lambda img: draw_line(img, (213, 338), (213, 100)),  # qd decrease line
        lambda img: write_text(img, "qd", (193, 60)),  # quantity axis
    ],
    tuple([0, 0, 0, 1, 0]): [  # tuple for quantity supplied increase
        lambda img: draw_line(img, (338, 338), (338, 100)),  # qs increase line
        lambda img: write_text(img, "qs", (318, 60)),  # quantity axis  
    ],
    tuple([0, 0, 0, 0, 1]): [  # tuple for surplus
        lambda img: draw_half_rectangle(img, (213, 400), (338, 375), (255, 255, 255), 1),  # Draw half-rectangle border
        lambda img: write_text(img, "Surplus", (220, 425))  # Surplus label
    ]
}

def execute(image, key):
    operations = pltm_dict.get(key)
    print(operations)
    if not operations:
        print(f"No operations found for key: {key}")
        return image
    for operation in operations:
        image = operation(image)
    return image

image = execute(image, tuple([0, 0, 1, 0, 0]))

# Convert image to RGB (if it's not already)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(image_rgb)
plt.title('Image with OpenCV Drawings')
plt.axis('off')
plt.show()
