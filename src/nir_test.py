import nir
import snntorch as snn
import torch
from snntorch.export_nir import export_to_nir

lif1 = snn.Leaky(beta=0.9, init_hidden=True)
lif2 = snn.Leaky(beta=0.9, init_hidden=True, output=True)

# Create a network
snntorch_network = torch.nn.Sequential(
    torch.nn.Flatten(), torch.nn.Linear(784, 500), lif1, torch.nn.Linear(500, 10), lif2
)

sample_data = torch.randn(1, 784)

# Export to nir
nir_graph = export_to_nir(snntorch_network, sample_data)
print(nir_graph.nodes)
print(nir_graph.edges)

new_graph = nir.NIRGraph(nodes=nir_graph.nodes, edges=nir_graph.edges)
# Save to file
nir.write("nir_model.nir", new_graph)
