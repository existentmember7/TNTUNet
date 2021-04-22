import json
import argparse
import matplotlib.pyplot as plt
import glob
from networkx.readwrite import json_graph
import networkx as nx
import numpy as np

parser = argparse.ArgumentParser(description='json file path')
parser.add_argument('--folder_path', default='', type=str, help='json file folder path')
args = parser.parse_args()

def main():
    data = None
    for file in glob.glob(args.folder_path+'*.json'):
        print("process file: "+file)
        with open(file) as f:
            data = json.load(f)

        intersection = []
        spacing = []

        for l in data['shapes']:

            if l['label'] == 'intersection':
                intersection.append(l['points'])
            elif l['label'] == 'spacing':
                spacing.append(l['points'])

        intersection = get_intersection_center(intersection)

        intersection_pair_list = []

        for s in spacing:
            pair = find_cloest_intersection(intersection, s)
            intersection_pair_list.append(pair)

        G = create_graph(intersection, intersection_pair_list)
        print(args.folder_path)
        show_graph(G)
        
        output_json(G, args.folder_path+file.split('/')[-1].split('.')[0]+"_new.json")
        exit(-1)

        
def show_graph(G):
        # draw edges by pts
        for (s,e) in G.edges():
            pt1 = G.nodes[s]['pts']
            pt2 = G.nodes[e]['pts']
            x = [pt1[0], pt2[0]]
            y = [pt1[1], pt2[1]]
            plt.plot(x, y, 'green', linewidth = 2)
            
        # draw node by o
        nodes = G.nodes()
        ps = np.array([nodes[i]['pts'] for i in nodes])
        print("node: ", ps)
        plt.plot(ps[:,0], ps[:,1], 'ro')
        count = 0
        for n in np.array([nodes[i] for i in nodes]):
            plt.annotate(count, (n['pts'][0], n['pts'][1]))
            count += 1

        # title and show
        plt.title('Build Graph')
        plt.show()

def create_graph(intersection, intersection_pair_list):
    G = nx.Graph()
    count = 0
    for i in intersection:
        G = add_node(G, count, i)
        count += 1
    for ip in intersection_pair_list:
        G = add_edge(G, ip[0], ip[1])
    return G

def add_node(G, index, pts):
    G.add_node(index)
    G.nodes[index]['pts'] = pts
    return G

def add_edge(G, node_index_1, node_index_2):
    G.add_edge(node_index_1, node_index_2)
    return G

def output_json(G, filename):
    data = json_graph.adjacency_data(G)
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)



def get_intersection_center(intersection_points):
    intersection = []
    for i in intersection_points:
        max_x = 0.0
        max_y = 0.0
        min_x = None
        min_y = None
        for p in i:
            if p[0] > max_x:
                max_x = p[0]
            if p[1] > max_y:
                max_y = p[1]
            if min_x == None or p[0] < min_x:
                min_x = p[0]
            if min_y == None or p[1] < min_y:
                min_y = p[1]
        intersection.append([(max_x+min_x)/2, (max_y+min_y)/2])
    return intersection

def find_cloest_intersection(intersection_pts, spacing_pts):
    min_distance_1 = None
    min_distance_2 = None
    count = 0
    pair_index_1 = None
    pair_index_2 = None
    for intersection_pt in intersection_pts:
        distance_1 = distance(intersection_pt,spacing_pts[0])
        distance_2 = distance(intersection_pt,spacing_pts[1])
        if min_distance_1 == None or distance_1 < min_distance_1:
            min_distance_1 = distance_1
            pair_index_1 = count
        if min_distance_2 == None or distance_2 < min_distance_2:
            min_distance_2 = distance_2
            pair_index_2 = count

        count += 1
    return [pair_index_1, pair_index_2]
        


def distance(pt1,pt2):
    return ((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**0.5
    
if __name__ == '__main__':
    main()