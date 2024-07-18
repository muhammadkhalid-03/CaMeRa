import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

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
    """
    Draws a line on the given image from the start point to the end point.

    Parameters:
    image (numpy.ndarray): The input image on which the line will be drawn.
    start (tuple): The (x, y) coordinates of the starting point of the line.
    end (tuple): The (x, y) coordinates of the ending point of the line.
    color (tuple, optional): The color of the line in RGB format. Default is white (255, 255, 255).
    thickness (int, optional): The thickness of the line in pixels. Default is 1.

    Returns:
    numpy.ndarray: The modified image with the drawn line.
    """
    start_adjusted = cart2cv(start)  # Adjust y-coordinate
    end_adjusted = cart2cv(end)  # Adjust y-coordinate
    return cv2.line(image, start_adjusted, end_adjusted, color, thickness)

# Helper function to write text (adjusted y-coordinate)
def write_text(image, text, position, color=(255, 255, 255), thickness=1):
    position_adjusted = cart2cv(position)  # Adjust y-coordinate
    return cv2.putText(image, text, position_adjusted, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, thickness, cv2.LINE_AA)

# Helper function to draw half-rectangle border (adjusted y-coordinates)
def draw_half_rectangle(image, top_left, bottom_right, color=(255, 255, 255), thickness=1):
    top_left_adjusted = cart2cv(top_left)  # Adjust y-coordinate
    bottom_right_adjusted = cart2cv(bottom_right)  # Adjust y-coordinate
    pts = np.array([[top_left_adjusted[0], bottom_right_adjusted[1]],
                    [top_left_adjusted[0], top_left_adjusted[1]],
                    [bottom_right_adjusted[0], top_left_adjusted[1]],
                    [bottom_right_adjusted[0], bottom_right_adjusted[1]]], np.int32)
    return cv2.polylines(image, [pts], isClosed=False, color=color, thickness=thickness)

#function to convert from cartesian coordinates to CV coordinates
def cart2cv(point):
    return (point[0], image.shape[0] - point[1])

def cv2cart(point):
    return (point[0], image.shape[0] - point[1])

# # Define the pltm_dict with image operations as lambdas
# pltm_dict = {
#     tuple([1, 0, 0, 0, 0]): [  # tuple for market equilibrium
#         lambda img: draw_line(img, (150, 400), (400, 150)),  # demand curve
#         lambda img: draw_line(img, (150, 150), (400, 400)),  # supply curve
#         lambda img: draw_line(img, (100, 100), (100, 450)),  # price axis
#         lambda img: draw_line(img, (100, 100), (450, 100)),  # quantity axis
#         lambda img: draw_line(img, (100, 275), (275, 275)),  # equilibrium price line
#         lambda img: draw_line(img, (275, 100), (275, 275)),  # equilibrium quantity line
#         lambda img: write_text(img, "P", (50, 450)),  # price axis label
#         lambda img: write_text(img, "Q", (450, 50)),  # quantity axis label
#         lambda img: write_text(img, "D", (120, 410)),  # demand curve label
#         lambda img: write_text(img, "S", (410, 410)),  # supply curve label
#         lambda img: write_text(img, "p1", (50, 265)),  # price 1 label
#         lambda img: write_text(img, "q", (255, 60)),  # price 2 label
#     ],
#     tuple([0, 1, 0, 0, 0]): [  # tuple for price increase
#         lambda img: draw_line(img, (100, 338), (338, 338)),  # price increase, qs line
#         lambda img: draw_line(img, (100, 338), (213, 338)),  # price increase, qd line
#         lambda img: write_text(img, "p2", (50, 330)),  # price axis label
#     ],
#     tuple([0, 0, 1, 0, 0]): [  # tuple for quantity demanded decrease
#         lambda img: draw_line(img, (213, 338), (213, 100)),  # qd decrease line
#         lambda img: write_text(img, "qd", (193, 60)),  # quantity axis
#     ],
#     tuple([0, 0, 0, 1, 0]): [  # tuple for quantity supplied increase
#         lambda img: draw_line(img, (338, 338), (338, 100)),  # qs increase line
#         lambda img: write_text(img, "qs", (318, 60)),  # quantity axis  
#     ],
#     tuple([0, 0, 0, 0, 1]): [  # tuple for surplus
#         lambda img: draw_half_rectangle(img, (213, 400), (338, 375)),  # Draw half-rectangle border
#         lambda img: write_text(img, "Surplus", (220, 425))  # Surplus label
#     ]
# }

# Define the pltm_dict with image operations as lambdas
pltm_dict = {
    tuple([1, 0, 0, 0, 0]): [  # tuple for market equilibrium
        ["line", (150, 400), (400, 150)],  # demand curve
        ["line", (150, 150), (400, 400)],  # supply curve
        ["line", (100, 100), (100, 450)],  # price axis
        ["line", (100, 100), (450, 100)],  # quantity axis
        ["line", (100, 275), (275, 275)],  # equilibrium price line
        ["line", (275, 100), (275, 275)],  # equilibrium quantity line
        ["label", "Quantity", (420, 60)],  # quantity axis label
        ["label", "Price", (20, 450)],  # price axis label
        ["label", "Demand", (120, 410)],  # demand curve label
        ["label", "Supply", (410, 410)],  # supply curve label
        ["label", "price 1", (20, 265)],  # price 1 label
        ["label", "qd, qs", (250, 60)],  # price 2 label
    ],
    tuple([0, 1, 0, 0, 0]): [  # tuple for price increase
        ["line", (100, 338), (338, 338)],  # price increase, qs line
        ["line", (100, 338), (213, 338)],  # price increase, qd line
        ["label", "price 2", (20, 330)],  # price axis label
    ],
    tuple([0, 0, 1, 0, 0]): [  # tuple for quantity demanded decrease
        ["line", (213, 338), (213, 100)],  # qd decrease line
        ["label", "qd", (193, 60)],  # quantity axis
    ],
    tuple([0, 0, 0, 1, 0]): [  # tuple for quantity supplied increase
        ["line", (338, 338), (338, 100)],  # qs increase line
        ["label", "qs", (330, 60)],  # quantity axis  
    ],
    tuple([0, 0, 0, 0, 1]): [  # tuple for surplus
        ["half-rectangle", (213, 400), (338, 375)],  # Draw half-rectangle border
        ["label", "Surplus", (230, 425)]  # Surplus label
    ]
}

def execute(image, key):
    operations = pltm_dict.get(key) #get value using key
    if not operations:
        print(f"No operations found for key: {key}")
        return image
    for operation in operations:
        match operation[0]:
            case "line":
                draw_line(image, operation[1], operation[2])
            case "half-rectangle":
                draw_half_rectangle(image, operation[1], operation[2])
            case "label":
                write_text(image, operation[1], operation[2])
    return image

image = execute(image, tuple([1, 0, 0, 0, 0]))
# for vec in pltm_dict.keys():
#     image = execute(image, vec)

cv2.imwrite('/Users/muhammadkhalid/Desktop/MAP/code/blackboard.png', image)
# Convert image to RGB (if it's not already)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(image_rgb)
plt.title('Image with OpenCV Drawings')
plt.axis('off')
plt.show()


# image = cv2.putText(image, 'OpenCV', (100, 50), cv2.FONT_HERSHEY_SIMPLEX,  
#                    1, (255, 255, 255), 2, cv2.LINE_AA) 
# plt.imshow(image)
# plt.show()
# # After creating and modifying your image
text = pytesseract.image_to_string(image, config='--psm 11')
print("Detected Text:")
print(text)












# Step 1: Preprocess the Image
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Step 2: Detect Lines Using Probabilistic Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=20, maxLineGap=10)
lines = lines.reshape(-1, 4)



def mergeLines(lines, threshold=50):
    """
    Merges duplicate lines in a list of lines.

    Parameters:
    lines (list): A list of lines, where each line is represented as a list of four integers.
    threshold (int, optional): The maximum allowed difference between the coordinates of two lines for them to be considered duplicates. Default is 2.

    Returns:
    list: A numpy array of merged lines, where each line is represented as a list of four integers.
    """

    if len(lines) == 0:
        return []
    
    arr = []
    arr = [lines[0]]
    for i in range(1, len(lines)):
        for j in range(i+1, len(lines)):
            if abs(lines[i][0]-lines[j][0])<=threshold and abs(lines[i][1]-lines[j][1])<=threshold and abs(lines[i][2]-lines[j][2])<=threshold and abs(lines[i][3]-lines[j][3])<=threshold:
                arr.append(lines[i])
    return arr




merged = mergeLines(lines)

#function to split points into array of tuples
def splitPoints(lines):
    return [[cv2cart((line[0], line[1])), cv2cart((line[2], line[3]))] for line in merged]

split_lines = splitPoints(merged)

for line in split_lines:
    print(line,'\n')



# Create a copy of the image to draw the lines
line_image = image.copy()

# Filter and draw detected lines
detected_lines = []
if lines is not None:
    for line in merged:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
        detected_lines.append((x1, y1, x2, y2))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)


# Display the image with detected lines
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Lines')
plt.axis('off')
plt.show()

# Function to calculate distance between two points
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Known coordinates for market equilibrium tuple
known_lines_market_eq = [
    ((150, 400), (400, 150)),
    ((150, 150), (400, 400)),
    ((100, 450), (100, 100)),
    ((100, 100), (450, 100)),
    ((100, 275), (275, 275)),
    ((275, 275), (275, 100))
]



# Step 3: Map Detected Lines to Tuples
def map_lines_to_tuple(detected_lines, known_lines):
    match_score = 0
    for d_line in detected_lines:
        for k_line in known_lines:
            dist1 = distance((d_line[0], d_line[1]), k_line[0])
            dist2 = distance((d_line[2], d_line[3]), k_line[1])
            if dist1 < 10 and dist2 < 10:  # Threshold for matching
                match_score += 1
    return match_score

# Compare detected lines with known lines for market equilibrium
market_eq_score = map_lines_to_tuple(detected_lines, known_lines_market_eq)

print(f"Market Equilibrium Match Score: {market_eq_score}")