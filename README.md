# Impact of Data Error on Phylogenetic Network Inference from Gene Trees Under the Multispecies Network Coalescent

## How to Use `generate_data.py`

This script is used to simulate phylogenetic data, including gene trees and sequence alignments, using external tools such as **ms**, **PhyloNet**, and **INDELible**.

### Requirements

Before using the script, make sure the following tools are installed and accessible from your system path:

- [`ms`](http://home.uchicago.edu/rhudson1/source/mksamples.html) — simulates gene trees under a coalescent model
- [`PhyloNet`](https://bioinfocs.rice.edu/phylonet) — performs species network simulations and phylogenetic network inference
- [`INDELible`](http://abacus.gene.ucl.ac.uk/software/indelible/) — simulates sequence evolution along trees with indels
- [`MAFFT`](https://mafft.cbrc.jp/alignment/software/) — for multiple sequence alignment
- [`IQ-TREE`](http://www.iqtree.org/) — for gene tree inference
- [`FastSP`](https://github.com/smirarab/FastSP) — for comparing tree topologies using split distance
- [`nw_reroot`](https://github.com/tjunier/newick_utils/tree/master) — reroots Newick trees
- [`nw_ed`](https://github.com/tjunier/newick_utils/tree/master) — edits Newick trees

Make sure all executables are installed and their paths are added to your `$PATH`, or adjust your script accordingly.

### Setup Instructions

You need to configure the script to point to the correct locations of the required tools. In `generate_data.py`, set the `root_folder` variable to the absolute path containing the tool directories and files. Then, make sure the following paths are correctly set:

```python
indelible_control_folder = root_folder + "INDELibleV1.03/"
phylonet = root_folder + "PhyloNet.jar"
ms_address = root_folder + "msdir/ms"
indelible_control_folder = root_folder + "INDELibleV1.03/"
iqtree_folder = root_folder + "iqtree/"
phylonet = root_folder + "PhyloNet.jar"
iqtree_pkg = root_folder + "iqtree-2.3.5-Linux-intel/bin/iqtree2"
mafft_pkg = "/shared/mt100/ml_env/bin/mafft"
mafft_result_addr = root_folder + "iqtree/result_mafft.txt"
iqtree_result_addr = root_folder + "iqtree/result.txt"
FastS_addr = root_folder + "FastSP/FastSP.jar"
```
## Inference and Plotting

To run the inference, execute the function `main_inference` in `main.py`:

## Plotting the Results

To plot the results, run the function `main_plotting` in `main.py`:

Because the functions run in parallel, make sure the address of `nw_reroot`, `nw_ed`, `result_addr`, and `FastSP` are correctly set in the functions `run_iqtree_bootstrap`, `run_iqtree`, and `run_mafft`.

The final_result.csv, which is a CSV file containing all the results, along with all the generated plots, can be found in the /`result`s folder.
