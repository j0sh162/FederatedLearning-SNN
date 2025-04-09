import nir
import numpy as np
import snntorch as snn
import torch
from rich import print
from snntorch.export_nir import export_to_nir

lif1 = snn.Leaky(beta=0.9, init_hidden=True)
lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

# Create a network
snntorch_network = torch.nn.Sequential(
    torch.nn.Flatten(), torch.nn.Linear(784, 500), lif1, torch.nn.Linear(500, 10), lif2
)

sample_data = torch.randn(1, 784, dtype=torch.float32)

# Export to nir
nir_graph = export_to_nir(module=snntorch_network.cpu(), sample_data=sample_data.cpu())

print("[red]Input type:", nir_graph.nodes["0"].input_type)
if nir_graph.nodes["0"].input_type is None:
    print("Input type is None")
    nir_graph.nodes["0"].input_type = {"input": np.array([784], dtype=np.int32)}

# nir_graph.edges = np.array(nir_graph.edges)
graph_dict = nir_graph.to_dict()
for key, value in graph_dict.items():
    print(key, type(value), value)

# Save to file
nir.write("nir_model.nir", nir_graph)

print("[red]LOADING GRAPH")

# Load from file
nir_graph = nir.read("nir_model.nir")
print("Loaded graph:", nir_graph)
print("Nodes:", nir_graph.nodes)
print("Edges:", nir_graph.edges)
