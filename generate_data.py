import os
import copy
import subprocess
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from collections import Counter
import numpy as np
import dendropy


def read_model_phylogeny(path):
    """
    Reads the phylogenetic network or species tree from a file.

    Parameters:
        path (str): Path to the file containing the Newick-formatted tree.

    Returns:
        str: Newick string with the model phylogeny.
    """
    with open(path, "r") as handle:
        content = handle.read()
    return content.strip()


def check_for_branch_zero(newick_str):
    """
    Checks if any edge in the tree has zero or negative branch length.

    Parameters:
        newick_str (str): Newick string representing a tree.

    Returns:
        bool: True if any branch length is zero, otherwise False.

    Raises:
        Exception: If a negative branch length is encountered.
    """
    tree = dendropy.Tree.get(data=newick_str, schema="newick")
    for edge in tree.postorder_edge_iter():
        if edge.length == 0.0:
            return True  # Found zero-length branch
        elif edge.length is not None and edge.length < 0:
            raise Exception("Negative branch length detected")
    return False

def scale_branch_length(genetrees_addr, genetrees_scaled_addr):
    """
    Scales the branch lengths of gene trees by a fixed multiplier and writes the result to a new file.

    Parameters:
        genetrees_addr (str): Path to the input file with unscaled gene trees (Newick format).
        genetrees_scaled_addr (str): Path to the output file for scaled gene trees.
    """
    # Load all gene trees from the input file
    trees = dendropy.TreeList.get(path=genetrees_addr, schema="newick")

    # Iterate through each tree and scale its branch lengths
    for tree in trees:
        for edge in tree.postorder_edge_iter():
            if edge.length is not None and edge.length > 0:
                # Scale the branch length by 0.018 and round to 6 decimal places
                edge.length = round(edge.length * 0.018, 6)
                assert edge.length > 0  # Ensure no zero-length branches were introduced
            elif edge.length is not None:
                # Raise an error if a negative or zero-length branch (non-null) is encountered
                raise Exception("Invalid edge length encountered")

    # Write the modified tree list to the output file
    trees.write(path=genetrees_scaled_addr, schema="newick", real_value_format_specifier=".6f")


def generate_gene_trees(addr_species, gt_folder, num_gt, phylonet, ms_address):
    # Read the model species network from file
    species = read_model_phylogeny(addr_species)

    # Save the species network to a .nw file
    with open(f"{gt_folder}species.nw", "w") as h:
        h.write(species)

    # Define file paths for intermediate and output files
    gen_gt_addr = f"{gt_folder}gen_gt.nex"
    genetrees_addr = f"{gt_folder}genetrees.txt"
    genetrees_scaled_addr = f"{gt_folder}genetrees_scaled.txt"

    # Create a NEXUS file that instructs PhyloNet to simulate gene trees
    nex_file = "#NEXUS\n" + f"BEGIN NETWORKS;\nNetwork net={species}\nEND;\n" + \
               f"\nBEGIN PHYLONET;\nSimGTInNetwork net {num_gt}" + f" -ms {ms_address}" + f";\n\nEND;\n "

    # Write the NEXUS instruction file
    with open(gen_gt_addr, "w") as h:
        h.write(nex_file)

    # Run PhyloNet to generate gene trees
    command = f"java -jar {phylonet} {gen_gt_addr}"
    print(command)
    result = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    gts_lst = result.strip().split("\n")[3:]  # Skip header lines

    # Run PhyloNet again to generate alternative trees for replacing zero-branch ones
    result1 = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
    result1_lst = result1.strip().split("\n")[3:]

    # Replace any trees that have zero-length branches
    j = 0
    for i in range(len(gts_lst)):
        flag = check_for_branch_zero(gts_lst[i])
        if flag:
            print(i)
            while True:
                if check_for_branch_zero(result1_lst[j]):
                    j += 1
                else:
                    j += 1
                    break
            gts_lst[i] = result1_lst[j]

    # Write the final gene trees to file
    with open(genetrees_addr, 'w') as f:
        f.write("\n".join(gts_lst))

    # Scale branch lengths and save the scaled trees
    scale_branch_length(genetrees_addr, genetrees_scaled_addr)


def generate_sequence_using_indelible(gt_folder, sites, indelible_control_folder, POWER_LAW_PARAMETER, INDEL_LENGTH,
                                      INDEL_RATE_LST, ALPHA_heterogeneity):
    # File paths for gene trees and output alignments
    gen_gt_addr = f"{gt_folder}genetrees.txt"
    genetrees_scaled_addr = f"{gt_folder}genetrees_scaled.txt"
    alignments_folder = gt_folder + "alignments/"

    # Create the alignment output directory if it doesn't exist
    os.makedirs(alignments_folder, exist_ok=True)

    # Ensure gene trees have scaled branch lengths
    scale_branch_length(gen_gt_addr, genetrees_scaled_addr)

    # Read scaled gene trees
    with open(genetrees_scaled_addr, "r") as handle:
        gen_gts = handle.read()

    # Load the control file template for INDELible
    with open(indelible_control_folder + "control_template.txt", "r") as handle:
        control_file_template = handle.read()

    # Set simulation seed and replace parameters in the template
    seed_value = 2478
    control_file_template = control_file_template.replace('RANDOM_SEED', str(seed_value))
    control_file_template = control_file_template.replace('POWER_LAW_PARAMETER', str(POWER_LAW_PARAMETER))
    control_file_template = control_file_template.replace('INDEL_LENGTH', str(INDEL_LENGTH))

    # Loop through each gene tree and indel rate to create control files and run simulations
    for i, gt in enumerate(gen_gts.strip().split("\n")):
        for INDEL_RATE in INDEL_RATE_LST:
            # Copy template and fill in simulation-specific parameters
            control_file_copy = copy.copy(control_file_template)
            OUTPUT_ADDR = f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_0"
            print(i, OUTPUT_ADDR)
            control_file_copy = control_file_copy.replace('INDEL_RATE', str(INDEL_RATE))
            control_file_copy = control_file_copy.replace('ALPHA_heterogeneity', str(ALPHA_heterogeneity))
            control_file_copy = control_file_copy.replace('NETWORK', str(gt))
            control_file_copy = control_file_copy.replace('SEQ_LENGTH', str(sites))
            control_file_copy = control_file_copy.replace('OUTPUT_ADDR', OUTPUT_ADDR)

            # Write the customized control file for INDELible
            with open(indelible_control_folder + "control.txt", "w") as handle:
                handle.write(control_file_copy + "\n")

            # Run INDELible using the control file
            command = f"cd {indelible_control_folder} && ./indelible"
            os.system(command)

            # Rename output files for clarity and consistency
            original_true_alignment_addr = f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_0_TRUE.fas"
            os.rename(original_true_alignment_addr,
                      f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_0_alignment_original.fas")

            original_without_alignment_addr = f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_0.fas"
            os.rename(original_without_alignment_addr,
                      f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_0_alignment_estimated.fas")


def add_error_to_alignmet(gt_folder, num_gt, error_rate, sites, alpha):
    # Define the path to the alignments directory
    alignments_folder = gt_folder + "alignments/"

    # Iterate over each gene tree alignment
    for i in range(num_gt):
        for INDEL_RATE in INDEL_RATE_LST:
            # Construct the path to the original (true) alignment file
            original_true_alignment_addr = f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_0_alignment_original.fas"
            assert os.path.isfile(original_true_alignment_addr)  # Ensure the file exists

            # Read the contents of the alignment file
            with open(original_true_alignment_addr, "r") as handle:
                content = handle.read()

            result = []
            for line in content.strip().split("\n"):
                if line.strip().startswith(f">"):
                    # Keep sequence headers unchanged
                    result.append(line)
                    continue
                else:
                    line = line.strip()

                    if error_rate == "repeat":
                        # Introduce repetition-based noise
                        repetition_counts = np.arange(1, 11)
                        dirichlet_probs = np.random.dirichlet([alpha] * 10)
                        err_rate = 0.005
                        vector = np.random.choice([False, True], size=len(line), p=[1 - err_rate, err_rate])
                        noisy_align = []
                        for idx, val in enumerate(vector):
                            if line[idx] == "-" or val == False:
                                noisy_align.append(line[idx])
                            else:
                                nucleotide = line[idx]
                                n = np.random.choice(repetition_counts, p=dirichlet_probs)
                                ACGT = [nucleotide] * n
                                noisy_align.append("".join(ACGT))
                        result.append("".join(noisy_align))

                    else:
                        # Introduce random substitution-based noise
                        vector = np.random.choice([False, True], size=len(line), p=[1 - error_rate, error_rate])
                        noisy_align = []
                        for idx, val in enumerate(vector):
                            if line[idx] == "-" or val == False:
                                noisy_align.append(line[idx])
                            else:
                                ACGT = np.random.choice(["A", "C", "G", "T"], size=1, p=[0.25, 0.25, 0.25, 0.25])
                                noisy_align.append(ACGT[0])
                        result.append("".join(noisy_align))

            result = "\n".join(result)

            # Define output file paths for alignments with introduced error
            erroneous_aligned_addr = f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_{error_rate}_alignment_original.fas"
            erroneous_not_aligned_addr = f"{alignments_folder}length_{sites}_alignment_{i}_indel_rate_{INDEL_RATE}_error_{error_rate}_alignment_estimated.fas"

            # Write erroneous original alignment (with gaps)
            with open(erroneous_aligned_addr, "w") as h:
                h.write(result)

            # Write erroneous estimated alignment (without gaps)
            with open(erroneous_not_aligned_addr, "w") as h:
                h.write(result.replace("-", ""))


def calculate_consensus(alignment: MultipleSeqAlignment) -> str:
    """Generates the consensus sequence from a multiple sequence alignment."""
    consensus = ''
    for i in range(alignment.get_alignment_length()):
        column = alignment[:, i]  # Extract column i (all characters at position i across sequences)
        most_common = Counter(column).most_common()  # Count character frequencies
        base = most_common[0][0]  # Select the most frequent character
        consensus += base  # Append to consensus sequence
    return consensus


def count_mismatches(alignment: MultipleSeqAlignment, consensus: str):
    """Counts mismatches of each sequence against the consensus."""
    mismatch_counts = {}
    for record in alignment:
        # Count mismatched characters between each sequence and the consensus
        mismatches = sum(1 for a, b in zip(record.seq, consensus) if a != b)
        mismatch_counts[record.id] = mismatches
    return mismatch_counts


def analysis_sequences(msa):
    # Read multiple sequence alignment in FASTA format
    alignment = AlignIO.read(msa, "fasta")

    # Calculate consensus sequence and mismatch dictionary
    consensus_seq = calculate_consensus(alignment)
    mismatch_dict = count_mismatches(alignment, consensus_seq)

    # Exclude the outgroup "OUT" from mismatch counting
    lst_species = list(mismatch_dict.keys())
    lst_species.remove("OUT")

    # Sum mismatches for all species except "OUT"
    count_mismatch = 0
    for specs in lst_species:
        count_mismatch += mismatch_dict[specs]

    return count_mismatch


if __name__ == '__main__':
    scenarios = ["s0_0_ret", "s1_1_ret_down", "s4_2_ret_cross"]
    num_tip_lst = [6]
    numbsim =  10
    start_replica = 0
    POWER_LAW_PARAMETER = 1.5
    INDEL_LENGTH = 5
    ALPHA_heterogeneity = 0.4
    INDEL_RATE_LST = [0, 0.05, 0.1]
    num_gt_lst = [100, 250, 500]
    error_rate_lst = [0.01, 0.1]
    sites_per_gt_lst = [200, 500, 1000]

    root_folder = "/shared/mt100/6_book_chapter_final/"
    indelible_control_folder = root_folder + "INDELibleV1.03/"
    alpha = 0.1
    poprate = 0.01  # we didn't use this
    phylonet= root_folder + "PhyloNet.jar"
    ms_address = root_folder + "msdir/ms"

    for scenario in scenarios:
        scenario_folder = f"{root_folder}{scenario}/"
        for i, num_species in enumerate(num_tip_lst):
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                os.makedirs(sim_folder, exist_ok=True)
                addr_species = scenario_folder + f"species.nw"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    os.makedirs(gt_folder, exist_ok=True)

                    # Gene tree generator
                    generate_gene_trees(addr_species, gt_folder, num_gt, phylonet, ms_address)
                    #sequence generator
                    for sites in sites_per_gt_lst:

                        generate_sequence_using_indelible(gt_folder, sites, indelible_control_folder, POWER_LAW_PARAMETER , INDEL_LENGTH , INDEL_RATE_LST, ALPHA_heterogeneity)
                        total_mismatches = 0
                        for INDEL_RATE in INDEL_RATE_LST:
                            for gt_idx in range(num_gt):
                                alignments_folder = gt_folder + "alignments/"
                                original_addr = f"{alignments_folder}length_{sites}_alignment_{gt_idx}_indel_rate_{INDEL_RATE}_error_0_alignment_original.fas"
                                count_mismatch = analysis_sequences(original_addr)
                                total_mismatches +=count_mismatch
                            mismatch_addr = f"{alignments_folder}mismatch_length_{sites}_indel_rate_{INDEL_RATE}_error_0_alignment_original.txt"
                            with open(mismatch_addr , "w") as handle:
                                handle.write(str(total_mismatches/num_gt) + "\n")
                        for error_rate in error_rate_lst:
                            add_error_to_alignmet(gt_folder, num_gt, error_rate, sites, alpha)

            print(f"Species {num_species} is finished")

