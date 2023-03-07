import tensorflow_gnn as tfgnn
import tensorflow as tf
import sklearn.model_selection
import numpy as np

import graph_tool.generation
import graph_tool.draw
import random


cycle_len = 10
data = []
random.seed(42)
for sample in range(100):
    g = graph_tool.generation.lattice([cycle_len,1])
    g.add_edge(g.vertex(0), g.vertex(cycle_len-1))

    v_spin = g.new_vertex_property("int")
    e_weight = g.new_edge_property("int")
    energy = g.new_graph_property("int")
    lattice_id = g.new_graph_property("int")

    for i in range(cycle_len):
        v = g.vertex(i)
        v_spin[v] = random.choice([-1, 1])
        if i > 0:
            e = g.edge(i-1, i)
            e_weight[e] = random.randrange(1, 11)
            energy[g] += e_weight[e] * v_spin[i] * v_spin[i-1]
    lattice_id[g] = sample

    g.vertex_properties["spin"] = v_spin
    g.edge_properties["weight"] = e_weight
    g.graph_properties["energy"] = energy
    g.graph_properties["lattice_id"] = lattice_id
    data.append(g)


def make_graph_tensor(graph):
    spin = graph.vertex_properties["spin"].get_array()
    weight = graph.edge_properties["weight"].get_array()
    energy = graph.graph_properties["energy"]
    lattice_id = graph.graph_properties["lattice_id"]

    source, target = [], []
    for e in graph.edges():
        source.append(graph.vertex_index[e.source()])
        target.append(graph.vertex_index[e.target()])
    

    particle = tfgnn.NodeSet.from_fields(features={'spin': np.array(spin)},
                                    sizes=tf.constant([1]))
    # print(particle)
    particle_adjacency = tfgnn.Adjacency.from_indices(source=('particle', source),
                                                  target=('particle', target))
    # print(particle_adjacency)
    bond = tfgnn.EdgeSet.from_fields(features={'weights': weight},
                                     adjacency=particle_adjacency,
                                     sizes=tf.constant([1]))
    # print(bond)
    context = tfgnn.Context.from_fields(features={'enegry': [energy], 'lattice': [lattice_id]})    
    return tfgnn.GraphTensor.from_pieces(node_sets={'particle': particle}, edge_sets={'bond': bond}, context=context)


def get_initial_map_features(hidden_size, activation='relu'):
    """
    Initial pre-processing layer for a GNN (use as a class constructor).
    """
    def node_sets_fn(node_set, node_set_name):
        if node_set_name == 'particle':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(node_set['spin'])
    
    def edge_sets_fn(edge_set, edge_set_name):
        if edge_set_name == 'bond':
            return tf.keras.layers.Dense(units=hidden_size, activation=activation)(edge_set['weights'])
    
    return tfgnn.keras.layers.MapFeatures(node_sets_fn=node_sets_fn,
                                          edge_sets_fn=edge_sets_fn,
                                          name='graph_embedding')


graph_schema_pbtxt = open("graph.pbtxt", 'r').read()
graph_schema = tfgnn.parse_schema(graph_schema_pbtxt)
graph_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

train_dataset = list(map(make_graph_tensor, data))
# print(train_dataset[0])
# # print(make_graph_tensor(`data[0]))
# print('--'*30)
# print(graph_spec.is_compatible_with(train_dataset[0]))
# print(graph_spec)

# print(train_dataset[0].edge_sets['bond'].adjacency.source)

graph_embedding = get_initial_map_features(hidden_size=128)
embedded_graph = graph_embedding(train_dataset)
embedded_graph.node_sets['atom'].features