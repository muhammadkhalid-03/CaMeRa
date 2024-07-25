import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cv2
import math
import pytesseract

class Memory:

    def __init__(self, vltm_nodes_dict, vltm_edges_list, pltm_dict):
        self.vltm_nodes_dict = vltm_nodes_dict
        self.vltm_edges_list = vltm_edges_list
        self.pltm_dict = pltm_dict
        self.graph = nx.DiGraph()
        
    #Function to intialize the knowledge graph in vLTM & vLTM
    def initializeKnowledgeGraph(self):
        for node, vector in self.vltm_nodes_dict.items():
            self.graph.add_node(node, vector=vector)
        self.graph.add_edges_from(self.vltm_edges_list)
        return self.graph
        
    def draw_graph(self, node_labels=True, edge_labels=False, node_color='lightblue', edge_color='gray', node_size=500, font_size=8, label_offset=0.05):
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_color, node_size=node_size)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_color)

        if node_labels:
            labels = {node: f"{node}\n{self.graph.nodes[node]['vector']}" for node in self.graph.nodes()}
            label_pos = {node: (x + label_offset, y) for node, (x, y) in pos.items()}
            nx.draw_networkx_labels(self.graph, label_pos, labels, font_size=font_size)

        if edge_labels:
            edge_labels = nx.get_edge_attributes(self.graph, 'label')
            if edge_labels:
                nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=font_size)

        plt.show()
        
        
        
        
        
class ImageProcessor:
    def __init__(self, memory):
        self.image = np.zeros((512, 512, 3), np.uint8)
        self.memory = memory

    def draw_line(self, start, end, color=(255, 255, 255), thickness=1):
        start_adjusted = self.cart2cv(start)
        end_adjusted = self.cart2cv(end)
        cv2.line(self.image, start_adjusted, end_adjusted, color, thickness)

    def write_text(self, text, position, color=(255, 255, 255), thickness=1):
        position_adjusted = self.cart2cv(position)
        cv2.putText(self.image, text, position_adjusted, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, thickness, cv2.LINE_AA)

    def draw_half_rectangle(self, top_left, bottom_right, color=(255, 255, 255), thickness=1):
        top_left_adjusted = self.cart2cv(top_left)
        bottom_right_adjusted = self.cart2cv(bottom_right)
        pts = np.array([[top_left_adjusted[0], bottom_right_adjusted[1]],
                        [top_left_adjusted[0], top_left_adjusted[1]],
                        [bottom_right_adjusted[0], top_left_adjusted[1]],
                        [bottom_right_adjusted[0], bottom_right_adjusted[1]]], np.int32)
        cv2.polylines(self.image, [pts], isClosed=False, color=color, thickness=thickness)

    def cart2cv(self, point):
        return (point[0], self.image.shape[0] - point[1])

    def cv2cart(self, point):
        return (point[0], self.image.shape[0] - point[1])

    def execute(self, key):
        operations = pltm_dict.get(key) #get value using key
        if not operations:
            print(f"No operations found for key: {key}")
            return self.image
        for operation in operations:
            match operation[0]:
                case "line":
                    self.draw_line(operation[1], operation[2])
                case "label":
                    self.write_text(operation[1], operation[2])
                case _:
                    self.execute(operation[0])
        return self.image

    #Function to draw the image in the memory
    def drawCurrentImage(self, path='/Users/muhammadkhalid/Desktop/MAP/code/blackboard.png'):
        # for vec in self.memory.pltm_dict.keys():
            # if vec != tuple([1, 1, 1, 1, 1]):
            #     image = self.execute(vec)
            #     print(vec)
        image = self.execute(tuple([1, 1, 1, 1, 1]))
        cv2.imwrite(path, image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title('Image with OpenCV Drawings')
        plt.axis('off')
        plt.show()
        return self.image
    
    def preprocess_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        return edges

    def display_image(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title('Image with OpenCV Drawings')
        plt.axis('off')
        plt.show()
        



    def merge_lines(self, lines, threshold=50):
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
        print(arr)
        return arr


    def split_points(self, lines):
        arr = [[self.cart2cv((line[0], line[1])), self.cart2cv((line[2], line[3]))] for line in lines]
        print("\nSplit:", arr)
        return [[self.cart2cv((line[0], line[1])), self.cart2cv((line[2], line[3]))] for line in lines]


    def parse_text(self, detected_items):
        pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
        text = pytesseract.image_to_data(self.image, config='--psm 11')
        for i, item in enumerate(text.splitlines()):
            if i == 0:
                continue
            item = item.split()
            if len(item) == 12:
                x, y, w, h = int(item[6]), int(item[7]), int(item[8]), int(item[9])
                x, y = self.cart2cv((x, y + h))
                detected_items.append(["label", item[11], (x, y)])
        return detected_items
    

    def detect_lines(self, detected_items):
        edges = self.preprocess_image()
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=5, maxLineGap=10)
        if lines is not None:
            lines = lines.reshape(-1, 4)
            merged = self.merge_lines(lines)
            lines = self.split_points(merged)
            line_image = np.zeros((512, 512, 3), np.uint8)
            for line in lines:
                cv2.line(line_image, self.cart2cv(line[0]), self.cart2cv(line[1]), (0, 0, 255), 2)
                detected_items.append(["line", line[0], line[1]])
            print("\nDetected:", detected_items) #working
        return detected_items

    
    def detect(self):
        detected_items = []
        detected_items = self.parse_text(detected_items)
        detected_items = self.detect_lines(detected_items)
        return detected_items


    def line_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    

    def match(self, dict, items):

        def flatten_operations(operations):
            flat_list = []
            for operation in operations:
                if isinstance(operation[0], tuple):
                    for sub_key in operation:
                        flat_list.extend(flatten_operations(dict[sub_key]))
                else:
                    flat_list.append(operation)
            return flat_list
        
        max_score = 0
        match_id = None
        length = len(items)
        for key, value in dict.items():
            flattened_value = flatten_operations(value)
            if len(flattened_value) == length:
                score = 0
                for i in range(length): #loop for detected items
                    print("\nSpecific item:", items[i])
                    for j in range(length): #loop for pltm_dict values
                        if items[i][0] == "line" and flattened_value[j][0] == "line":
                            dist1 = math.dist(items[i][1], flattened_value[j][1])
                            dist2 = math.dist(items[i][2], flattened_value[j][2])
                            if dist1 < 10 and dist2 < 10:
                                score += 1
                        elif items[i][0] == "label" and flattened_value[j][0] == "label":
                            if math.dist(items[i][2], flattened_value[j][2]) < 10:
                                score += 1
                if score > max_score:
                    max_score = score
                    match_id = key
        return (match_id, dict[match_id])

    
if __name__ == "__main__":
    
    vltm_nodes = {
        "market_equilibrium": np.array([1, 0, 0, 0, 0]), 
        "price_increase": np.array([0, 1, 0, 0, 0]),
        "qd_decrease": np.array([0, 0, 1, 0, 0]), 
        "qs_increase": np.array([0, 0, 0, 1, 0]),
        "surplus": np.array([0, 0, 0, 0, 1]),
        "market_disequilibrium": np.array([1, 1, 1, 1, 1])
    }
    
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
            ["label", "Demand", (120, 420)],  # demand curve label
            ["label", "Supply", (410, 420)],  # supply curve label
            ["label", "price-1", (20, 265)],  # price 1 label
            ["label", "qd-qs", (250, 60)],  # price 2 label
        ],
        tuple([0, 1, 0, 0, 0]): [  # tuple for price increase
            ["line", (100, 338), (338, 338)],  # price increase, qs line
            ["label", "price-2", (20, 330)],  # price axis label
        ],
        tuple([0, 0, 1, 0, 0]): [  # tuple for quantity demanded decrease
            ["line", (213, 100), (213, 338)],  # qd decrease line
            ["label", "qd", (193, 60)],  # quantity axis
        ],
        tuple([0, 0, 0, 1, 0]): [  # tuple for quantity supplied increase
            ["line", (338, 100), (338, 338)],  # qs increase line
            ["label", "qs", (330, 60)],  # quantity axis  
        ],
        tuple([0, 0, 0, 0, 1]): [  # tuple for surplus
            ["label", "Surplus", (230, 350)]  # Surplus label
        ],
        tuple([1, 1, 1, 1, 1]): [ # tuple for market disequilibrium
            [tuple([1, 0, 0, 0, 0])],
            [tuple([0, 1, 0, 0, 0])],
            [tuple([0, 0, 1, 0, 0])],   
            [tuple([0, 0, 0, 1, 0])],    
            [tuple([0, 0, 0, 0, 1])]
        ]
    }
    
    vltm_edges = [
            ("market_equilibrium", "price_increase"),
            ("price_increase", "qd_decrease"),
            ("price_increase", "qs_increase"),
            ("qd_decrease", "surplus"),
            ("qs_increase", "surplus"),
            ("surplus", "market_disequilibrium")
        ]
    
    
    memory = Memory(vltm_nodes, vltm_edges, pltm_dict)
    memory.initializeKnowledgeGraph()
    memory.draw_graph()
    
    image = ImageProcessor(memory)
    image.drawCurrentImage()
    detected_items = image.detect()
    print(image.match(memory.pltm_dict, detected_items))
    
