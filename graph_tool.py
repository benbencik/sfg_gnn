import graph_tool.generation
import graph_tool.draw
import random

cycle_len = 10
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

# pos = graph_tool.draw.sfdp_layout(g, cooling_step=0.95, epsilon=1e-2)
# graph_tool.draw.graph_draw(g, vertex_text=v_spin, edge_text=e_weight, output="lattice.pdf")