import subprocess
import time
import copy
from ete3 import Tree
import math

import dendropy
from dendropy.calculate import treecompare
import os.path
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from multiprocessing import Process
import os
import psutil
import socket
import itertools


from collections import Counter


def run_mafft(args, cpu_id):
    result_addr = "/shared/mt100/6_book_chapter_final/iqtree/result_mafft.txt"
    pid = os.getpid()  # Get the current process ID
    os.sched_setaffinity(pid, {cpu_id})  # Set CPU affinity to bind the process to a specific core
    lst_result = []
    for command in args:
        # Run MAFFT alignment command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        stdout, stderr = process.communicate()

        # Extract MAFFT output file path
        mafft_addr = command.split(" ")[-1]
        assert os.path.isfile(mafft_addr)  # Ensure the output file exists

        # Get path to the original unmodified alignment
        original_addr = mafft_addr.split("estimated")[0] + "original.fas"
        assert os.path.isfile(original_addr)

        # Compute alignment distance metrics using FastSP
        command_distance_ca = f"/shared/mt100/ml_env/bin/java -jar /shared/mt100/6_book_chapter_final/FastSP/FastSP.jar -r {original_addr} -e {mafft_addr}"
        process1 = subprocess.Popen(command_distance_ca, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        distances, stderr1 = process1.communicate()
        distances = distances.strip().split("\n")

        # Extract SP and TC scores from FastSP output
        sp_score = float(distances[0].split(" ")[1].strip())
        tc_score = float(distances[6].split(" ")[1].strip())

        # Parse timing information from stderr
        time_command = stderr.split("Command being timed")[1]
        time_command = time_command.strip().split("\n")
        assert "User time" in time_command[1]
        assert "System time" in time_command[2]
        user_time = float(time_command[1].split(":")[1].strip())
        system_time = float(time_command[2].split(":")[1].strip())
        elased_time = str(user_time + system_time)

        # Append results in custom format
        lst_result.append(mafft_addr + "::" + elased_time + "::" + str(sp_score) +  "::" + str(tc_score) + "\n")

    # Write results to output file
    with open(result_addr, "a") as h:
        h.write("\n".join(lst_result))
    print(f"Reported: {len(args)} mafft, cpu_id:{cpu_id}")

def config_mafft(scenarios, root_folder, mafft_pkg, iqtree_folder, mafft_result_addr, INDEL_RATE_LST, error_rate_lst, gap_penalty, cpu_cores, numbsim, start_replica ):
    os.makedirs(iqtree_folder, exist_ok=True)  # Ensure output folder exists
    for scenario in scenarios:
        scenario_folder = f"{root_folder}{scenario}/"
        for i, num_species in enumerate(num_tip_lst):
            if os.path.isfile(mafft_result_addr):
                os.remove(mafft_result_addr)  # Clear old results

            commands = []
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    alignments_folder = gt_folder + "alignments/"
                    for sites in sites_per_gt_lst:
                        for j in range(num_gt):
                            for error_rate in error_rate_lst:
                                for indel_rate in INDEL_RATE_LST:
                                    # Construct input and output paths for MAFFT
                                    erroneous_not_aligned_addr = f"{alignments_folder}length_{sites}_alignment_{j}_indel_rate_{indel_rate}_error_{error_rate}_alignment_estimated.fas"
                                    output_addr = erroneous_not_aligned_addr.split((".fas"))[0] + "_mafft.fas"
                                    # Create MAFFT command
                                    command = f"/usr/bin/time -v {mafft_pkg} --globalpair --maxiterate 1000  --ep {gap_penalty} {erroneous_not_aligned_addr} > {output_addr}"
                                    commands.append(command)
            # Run MAFFT commands in parallel
            run_parallel(commands, run_mafft, max_parallel_processes=cpu_cores, batch_size=round(len(commands)/cpu_cores) + 1)
            # Parse and store MAFFT results
            parse_mafft_result(scenario_folder, mafft_result_addr, num_species, numbsim, start_replica, num_gt_lst, sites_per_gt_lst, INDEL_RATE_LST, error_rate_lst)
            print(f"Species {num_species} is finished")

def parse_mafft_result(scenario_folder, mafft_result_addr, num_species, numbsim, start_replica, num_gt_lst, sites_per_gt_lst, INDEL_RATE_LST, error_rate_lst):
    result_dict = {}
    # Read MAFFT results from file
    with open(mafft_result_addr, "r") as h:
        result = h.read()
    result = result.split()
    for line in result:
        addr, elased_time, sp_score, tc_score = line.split("::")
        result_dict[addr] = [elased_time, sp_score, tc_score]

    # Store metrics for each alignment
    for sim in range(start_replica, numbsim):
        sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
        for num_gt in num_gt_lst:
            gt_folder = sim_folder + f"{num_gt}_gene_trees/"
            for sites in sites_per_gt_lst:
                alignments_folder = gt_folder + "alignments/"
                mafft_folder = gt_folder + f"maffts/"
                os.makedirs(mafft_folder, exist_ok=True)
                for error_rate in error_rate_lst:
                    for indel_rate in INDEL_RATE_LST:
                        # Output file paths for scores and timing
                        mafft_file_addr = mafft_folder + f"mafft_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}"
                        mafft_times_addr = f"{mafft_file_addr}_times"
                        mafft_sp_scores_addr =  f"{mafft_file_addr}_sp_scores"
                        mafft_tc_scores_addr =  f"{mafft_file_addr}_tc_scores"

                        lst_sp_score = []
                        lst_tc_score = []
                        lst_times = []

                        for j in range(num_gt):
                            file_temp = f"{alignments_folder}length_{sites}_alignment_{j}_indel_rate_{indel_rate}_error_{error_rate}_alignment_estimated_mafft.fas"
                            elased_time, sp_score, tc_score = result_dict[file_temp]
                            lst_sp_score.append(sp_score)
                            lst_tc_score.append(tc_score)
                            lst_times.append(elased_time)

                        # Write SP scores, TC scores, and times to files
                        with open(mafft_sp_scores_addr, "w") as h:
                            h.write("\n".join(lst_sp_score))
                        with open(mafft_tc_scores_addr, "w") as h:
                            h.write("\n".join(lst_tc_score))
                        with open(mafft_times_addr, "w") as h:
                            h.write("\n".join(lst_times))


def run_iqtree(args, cpu_id):
    result_addr = "/shared/mt100/6_book_chapter_final/iqtree/result.txt"
    pid = os.getpid()  # Get the current process ID
    os.sched_setaffinity(pid, {cpu_id})  # Bind the process to a specific CPU core
    lst_result = []
    for command in args:
        # Run IQ-TREE command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        stdout, stderr = process.communicate()

        # Extract alignment file path from command
        alignment_addr = command.split(" ")[4]
        tree_addr = alignment_addr + ".treefile"
        assert os.path.isfile(tree_addr)  # Ensure the tree file exists

        # Read inferred unrooted tree
        with open(tree_addr, "r") as h:
            infered_tree = h.read()

        # Root the inferred tree using specified outgroup
        command_rooting = f"/shared/mt100/ml_env/bin/nw_reroot {tree_addr} OUT"
        process1 = subprocess.Popen(command_rooting, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        rooted_tree, stderr1 = process1.communicate()
        assert stderr1 == ""  # Ensure no error in rooting
        rooted_tree = rooted_tree.strip()

        # Parse timing information from stderr
        time_command = stderr.strip().split("\n")
        assert "User time" in time_command[1]
        assert "System time" in time_command[2]
        user_time = float(time_command[1].split(":")[1].strip())
        system_time = float(time_command[2].split(":")[1].strip())
        elased_time = str(user_time + system_time)

        # Append result to list
        lst_result.append(alignment_addr + "::" + rooted_tree.strip() + "::" + elased_time + "::" +  infered_tree + "\n")

        # Clean up intermediate files generated by IQ-TREE
        os.remove(alignment_addr + ".bionj")
        os.remove(alignment_addr + ".ckp.gz")
        os.remove(alignment_addr + ".iqtree")
        os.remove(alignment_addr + ".log")
        os.remove(alignment_addr + ".mldist")
        os.remove(alignment_addr + ".model.gz")
        os.remove(alignment_addr + ".treefile")

    # Write results to file
    with open(result_addr, "a") as h:
        h.write("\n".join(lst_result))
    print(f"Reported: {len(args)} iqtree, cpu_id:{cpu_id}")

def config_iqtree(scenarios, root_folder, iqtree_pkg, iqtree_folder, error_rate_lst, iqtree_result_addr, INDEL_RATE_LST, cpu_cores, numbsim, start_replica):
    os.makedirs(iqtree_folder, exist_ok=True)  # Create output folder if not exists
    for scenario in scenarios:
        if os.path.isfile(iqtree_result_addr):
            os.remove(iqtree_result_addr)  # Remove previous results
        scenario_folder = f"{root_folder}{scenario}/"
        commands = []
        for i, num_species in enumerate(num_tip_lst):
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    alignments_folder = gt_folder + "alignments/"
                    for sites in sites_per_gt_lst:
                        for indel_rate in INDEL_RATE_LST:
                            for error_rate in error_rate_lst:
                                for j in range(num_gt):
                                    for erroneous in ["original", "estimated_mafft"]:
                                        # Skip redundant original if repeated error model
                                        if error_rate == "repeat" and erroneous == "original":
                                            continue
                                        # Construct alignment file path
                                        addr_seq = f"{alignments_folder}length_{sites}_alignment_{j}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.fas"
                                        assert os.path.isfile(addr_seq)  # Ensure alignment file exists

                                        # Build IQ-TREE command
                                        model = "MFP"
                                        type_of_seq = "DNA"
                                        command = f"/usr/bin/time -v {iqtree_pkg} -s {addr_seq} -m {model} -st {type_of_seq} -redo"
                                        commands.append(command)

        # Run IQ-TREE in parallel on multiple CPUs
        run_parallel(commands, run_iqtree, max_parallel_processes=cpu_cores, batch_size=round(len(commands)/cpu_cores) + 1)

        # Parse and save IQ-TREE outputs
        parse_iqtree_result(scenario_folder, iqtree_result_addr, num_species, numbsim, start_replica, num_gt_lst, sites_per_gt_lst, error_rate_lst, INDEL_RATE_LST)
        print(f"Species {num_species} is finished")

def parse_iqtree_result(scenario_folder, iqtree_result_addr, num_species, numbsim, start_replica, num_gt_lst,
                        sites_per_gt_lst, error_rate_lst, INDEL_RATE_LST):
    result_dict = {}
    # Read and split raw result file
    with open(iqtree_result_addr, "r") as h:
        result = h.read()
    result = result.split()

    # Parse result lines into dictionary
    for line in result:
        addr, rooted_tree, elased_time, unrooted_tree = line.split("::")
        result_dict[addr] = [rooted_tree, elased_time, unrooted_tree]

    # For each simulation, write rooted/unrooted trees and timings
    for sim in range(start_replica, numbsim):
        sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
        for num_gt in num_gt_lst:
            gt_folder = sim_folder + f"{num_gt}_gene_trees/"
            iqtree_folder = gt_folder + f"iqtrees/"
            os.makedirs(iqtree_folder, exist_ok=True)
            alignments_folder = gt_folder + "alignments/"
            for sites in sites_per_gt_lst:
                for error_rate in error_rate_lst:
                    for indel_rate in INDEL_RATE_LST:
                        for erroneous in ["original", "estimated_mafft"]:
                            if error_rate == "repeat" and erroneous == "original":
                                continue

                            # Define output paths for each metric
                            iqtree_rooted_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_rooted"
                            iqtree_unrooted_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_unrooted"
                            iqtree_times_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_times"

                            lst_rooted_trees = []
                            lst_unrooted_trees = []
                            lst_times = []

                            for j in range(num_gt):
                                # Build alignment key and extract stored metrics
                                addr_res = f"{alignments_folder}length_{sites}_alignment_{j}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.fas"
                                rooted_tree, elased_time, unrooted_tree = result_dict[addr_res]
                                lst_rooted_trees.append(rooted_tree)
                                lst_unrooted_trees.append(unrooted_tree)
                                lst_times.append(elased_time)

                            # Write per-alignment results to files
                            with open(iqtree_rooted_addr, "w") as h:
                                h.write("\n".join(lst_rooted_trees))
                            with open(iqtree_unrooted_addr, "w") as h:
                                h.write("\n".join(lst_unrooted_trees))
                            with open(iqtree_times_addr, "w") as h:
                                h.write("\n".join(lst_times))



def run_iqtree_bootstrap(args, cpu_id):
    result_addr = "/shared/mt100/6_book_chapter_final/iqtree/result.txt"
    pid = os.getpid()  # Get the current process ID
    os.sched_setaffinity(pid, {cpu_id})  # Set affinity to the specified CPU core
    lst_result = []
    for command in args:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        stdout, stderr = process.communicate()

        alignment_addr = command.split(" ")[4]
        tree_addr = alignment_addr + ".treefile"
        assert os.path.isfile(tree_addr)

        with open(tree_addr, "r") as h:
            infered_tree = h.read()


        command_collapsing = f"/shared/mt100/ml_env/bin/nw_ed {tree_addr} 'i & b <= 70' o "
        process2 = subprocess.Popen(command_collapsing, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        collapsed_tree, stderr2 = process2.communicate()
        assert  stderr2==""
        collapsed_tree = collapsed_tree.strip()
        with open(tree_addr, "w") as h:
            h.write(collapsed_tree + '\n')
        command_rooting = f"/shared/mt100/ml_env/bin/nw_reroot {tree_addr} OUT"
        process1 = subprocess.Popen(command_rooting, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        rooted_collapsed_tree, stderr1 = process1.communicate()
        assert  stderr1==""
        rooted_collapsed_tree = rooted_collapsed_tree.strip()

        time_command = stderr.strip().split("\n")
        assert "User time" in time_command[1]
        assert "System time" in time_command[2]
        user_time = float(time_command[1].split(":")[1].strip())
        system_time = float(time_command[2].split(":")[1].strip())
        elased_time = str(user_time + system_time)
        lst_result.append(alignment_addr + "::" + rooted_collapsed_tree.strip() + "::" + elased_time + "::" +  infered_tree +"\n")
        os.remove(alignment_addr + ".bionj")
        os.remove(alignment_addr + ".ckp.gz")
        os.remove(alignment_addr + ".iqtree")
        os.remove(alignment_addr + ".log")
        os.remove(alignment_addr + ".mldist")
        os.remove(alignment_addr + ".model.gz")
        os.remove(alignment_addr + ".treefile")
        os.remove(alignment_addr + ".splits.nex")
        os.remove(alignment_addr + ".contree")
    with open(result_addr, "a") as h:
        h.write("\n".join(lst_result))
    print(f"Reported: {len(args)} iqtree, cpu_id:{cpu_id}")

def config_iqtree_bootstrap(scenarios, root_folder, iqtree_pkg, iqtree_folder, error_rate_lst, iqtree_result_addr, INDEL_RATE_LST, cpu_cores, numbsim, start_replica):
    os.makedirs(iqtree_folder, exist_ok=True)
    for scenario in scenarios:
        if os.path.isfile(iqtree_result_addr):
            os.remove(iqtree_result_addr)
        scenario_folder = f"{root_folder}{scenario}/"
        commands = []
        for i, num_species in enumerate(num_tip_lst):
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    alignments_folder = gt_folder + "alignments/"
                    for sites in sites_per_gt_lst:
                        # batch = []
                        for indel_rate in INDEL_RATE_LST:
                            for error_rate in error_rate_lst:
                                for j in range(num_gt):
                                    for erroneous in ["original", "estimated_mafft"]:
                                        addr_seq = f"{alignments_folder}length_{sites}_alignment_{j}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.fas"
                                        assert os.path.isfile(addr_seq)
                                        model = "MFP"
                                        type_of_seq = "DNA"
                                        command = f"/usr/bin/time -v {iqtree_pkg} -s {addr_seq} -m {model} -st {type_of_seq} -B 1000 -redo"
                                        commands.append(command)
        run_parallel(commands, run_iqtree_bootstrap, max_parallel_processes=cpu_cores, batch_size=round(len(commands)/cpu_cores) + 1)
        parse_iqtree_bootstrap_result(scenario_folder, iqtree_result_addr, num_species, numbsim, start_replica, num_gt_lst, sites_per_gt_lst, error_rate_lst, INDEL_RATE_LST)
        print(f"Species {num_species} is finished")


def parse_iqtree_bootstrap_result(scenario_folder, iqtree_result_addr, num_species, numbsim, start_replica, num_gt_lst, sites_per_gt_lst, error_rate_lst, INDEL_RATE_LST):
    result_dict = {}
    with open(iqtree_result_addr, "r") as h:
        result = h.read()
    result = result.split()
    for line in result:
        addr, rooted_collapsed_tree, elased_time, unrooted_tree= line.split("::")
        result_dict[addr] = [rooted_collapsed_tree, elased_time, unrooted_tree]
    for sim in range(start_replica, numbsim):
        sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
        for num_gt in num_gt_lst:
            gt_folder = sim_folder + f"{num_gt}_gene_trees/"
            iqtree_folder = gt_folder + f"iqtrees/"
            os.makedirs(iqtree_folder, exist_ok=True)
            alignments_folder = gt_folder + "alignments/"
            for sites in sites_per_gt_lst:
                for error_rate in error_rate_lst:
                    for indel_rate in INDEL_RATE_LST:
                        for erroneous in ["original", "estimated_mafft"]:
                            iqtree_rooted_collapsed_addr = iqtree_folder + f"bootstrap_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_rooted"
                            iqtree_unrooted_addr = iqtree_folder + f"bootstrap_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_unrooted"
                            iqtree_times_addr = iqtree_folder + f"bootstrap_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_times"
                            lst_rooted_collapsed_trees = []
                            lst_unrooted_trees = []
                            lst_times = []
                            for j in range(num_gt):
                                addr_res = f"{alignments_folder}length_{sites}_alignment_{j}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.fas"
                                rooted_collapsed_tree, elased_time, unrooted_tree = result_dict[addr_res]
                                lst_rooted_collapsed_trees.append(rooted_collapsed_tree)
                                lst_unrooted_trees.append(unrooted_tree)
                                lst_times.append(elased_time)

                            with open(iqtree_rooted_collapsed_addr, "w") as h:
                                h.write("\n".join(lst_rooted_collapsed_trees))
                            with open(iqtree_unrooted_addr, "w") as h:
                                h.write("\n".join(lst_unrooted_trees))
                            with open(iqtree_times_addr, "w") as h:
                                h.write("\n".join(lst_times))


def calc_different_distance(estimated_network, target_network, results_folder):
    if "#" not in estimated_network and "#" not in target_network:  # if both networks are tree then calculate RF distance
        distance_RF = calc_RF_distance(estimated_network, target_network)
        num_inferred_reticulations = 0
        num_real_reticulations = 0
        distance_luay, distance_rnbs, distance_normwapd = "", "", ""

    else:
        count = estimated_network.count("#")
        assert count % 2 == 0
        num_inferred_reticulations = int(count / 2)
        count = target_network.count("#")
        assert count % 2 == 0
        num_real_reticulations = int(count / 2)
        distance_luay, distance_rnbs, distance_normwapd = calc_net_distance( estimated_network, target_network, results_folder)
        distance_RF = ""
    return distance_RF, distance_luay, distance_rnbs, distance_normwapd, num_inferred_reticulations, num_real_reticulations


def run_infer_net(args, cpu_id):
    """Run the command and append the output to the specified file."""
    for command, infer_net_out_addr, gt_folder in args:
        pid = os.getpid()  # Get the current process ID
        os.sched_setaffinity(pid, {cpu_id})  # Set affinity to the specified CPU core
        print(f"core_{cpu_id}:{command}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        stdout, stderr = process.communicate()
        addr_species = f"{gt_folder}species.nw"
        with open(addr_species, "r") as h:
            net = h.read()

        # remove first line
        result = stdout.strip().split("\n")[1:]
        method = infer_net_out_addr.split("/")[-2]
        assert method in ['InferNetwork_MPL', 'InferNetwork_ML', 'MCMC_GT', 'MCMC_GT_pseudo']
        if  method =='MCMC_GT' or method=='MCMC_GT_pseudo':
            lst_estimations = []
            for idx, elm in enumerate(result):
                if elm.startswith("Rank ="):
                    rec = result[idx].split(";")
                    percentage = float(rec[2].split("=")[1].strip())
                    prob = float(rec[3].split(":")[0].split("=")[1].strip())
                    estimated_especies = ":".join(rec[3].split(":")[1:]) + ";"
                    lst_estimations.append([percentage, prob, estimated_especies])
            assert lst_estimations[0][0] == max([i[0] for i in lst_estimations])
            percent, prob, estim = lst_estimations[0]
            distance_RF, distance_luay, distance_rnbs, distance_normwapd, num_inferred_reticulations, num_real_reticulations = calc_different_distance(estim, net, gt_folder)
            result.insert(len(result), f"\nEstimated_network:\t {estim} \nTarget_network:\t {net}\nPercentage: {percent}\nLog Probability: {prob}\ndistance_RF: {distance_RF}\ndistance_luay: {distance_luay}\ndistance_rnbs: {distance_rnbs}\ndistance_normwapd: {distance_normwapd}\nnum_inferred_reticulations: {num_inferred_reticulations}\nnum_real_reticulations: {num_real_reticulations}\n")
        elif method == 'InferNetwork_MPL' or method == 'InferNetwork_ML':
            lst_estimations = []
            for idx, elm in enumerate(result):
                if elm.startswith("Inferred Network "):
                    num_net = int(result[idx].split("#")[1][0:-1])
                    prob = float(result[idx+2].split(":")[1].strip("Running time"))
                    estimated_species = result[idx+1]
                    lst_estimations.append([prob, estimated_species])
            assert lst_estimations[0][0] == max([i[0] for i in lst_estimations])
            prob, estim = lst_estimations[0]
            distance_RF, distance_luay, distance_rnbs, distance_normwapd, num_inferred_reticulations, num_real_reticulations = calc_different_distance(estim, net, gt_folder)
            result.insert(len(result), f"\nEstimated_network:\t {estim} \nTarget_network:\t {net}\nLog Probability: {prob}\ndistance_RF: {distance_RF}\ndistance_luay: {distance_luay}\ndistance_rnbs: {distance_rnbs}\ndistance_normwapd: {distance_normwapd}\nnum_inferred_reticulations: {num_inferred_reticulations}\nnum_real_reticulations: {num_real_reticulations}\n")

        result1 = "\n".join(result)
        result1 += "\n"
        result1 += stderr
        with open(infer_net_out_addr, "w") as h:
            h.write(result1)
        print(f"Reported: {infer_net_out_addr}")


def run_parallel(commands, target_func, max_parallel_processes, batch_size):
    active_processes = []
    threshold_percentage = 10.0
    enough_cores = False
    num_batches = len(commands) / batch_size
    if round(num_batches) == num_batches:
        num_batches = int(num_batches)
    else:
        num_batches = int(num_batches) + 1
    max_parallel_processes = min(num_batches, max_parallel_processes)

    while enough_cores == False:
        # Get the CPU utilization for each core
        cpu_percentages = psutil.cpu_percent(percpu=True, interval=1)
        free_cpus = [i for i, usage in enumerate(cpu_percentages) if usage < threshold_percentage]
        if len(free_cpus) > max_parallel_processes:
            enough_cores = True
        else:
            print(f"Number of Cores is {len(free_cpus)} while we need {max_parallel_processes}")
            time.sleep(1)
    lst_cores_ids = free_cpus[0:max_parallel_processes]
    batches = []
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(commands[start:end])
    print(f"Number of batches: {num_batches}, size each batches: {batch_size}, Number of commands: {len(commands)}")
    for args in batches:
        while len(active_processes) >= max_parallel_processes:
            # Check if any process has finished
            for process, id in active_processes:
                if not process.is_alive():
                    cpu_id_removed = id
                    active_processes.remove([process, id])
                else:
                    time.sleep(1)  # Sleep for a short time before checking again
        temp_cores_ids = copy.deepcopy(lst_cores_ids)
        for process, id in active_processes:
            temp_cores_ids.remove(id)
        assert len(temp_cores_ids) > 0
        cpu_id = temp_cores_ids[0]
        process = Process(target=target_func, args=(args, cpu_id))
        active_processes.append([process, cpu_id])
        process.start()
        time.sleep(0.1)

    # Wait for all processes to finish
    for process, id in active_processes:
        process.join()


def config_infer_net(root_folder, scenarios, phylonet, num_tip_lst, numbsim, start_replica, num_gt_lst, lst_max_reticulation, error_rate_lst, INDEL_RATE_LST, methods, params_mcmc_gt, cpu_cores):
    commands = []

    for scenario in scenarios:
        scenario_folder = f"{root_folder}{scenario}/"
        for i, num_species in enumerate(num_tip_lst):
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    for method in methods:
                        method_folder = gt_folder +  f"{method}/"
                        os.makedirs(method_folder, exist_ok=True)
                        for max_reticulation in lst_max_reticulation:
                            for iqtree_flag in [False, True]:
                                if iqtree_flag==False:
                                        command, infer_net_out_addr = produce_commands_infer_net(phylonet, gt_folder, method_folder, num_gt, iqtree_flag, 0, max_reticulation, method, params_mcmc_gt, 0, 0, 0)
                                        if command != "":
                                            commands.append((command, infer_net_out_addr, gt_folder))
                                else:
                                    for sites in sites_per_gt_lst:
                                        for error_rate in error_rate_lst:
                                            for indel_rate in INDEL_RATE_LST:
                                                for erroneous in ["original", "estimated_mafft"]:
                                                    if error_rate == "repeat" and erroneous == "original":
                                                        continue
                                                    command, infer_net_out_addr = produce_commands_infer_net(phylonet, gt_folder, method_folder, num_gt, iqtree_flag, sites, max_reticulation, method, params_mcmc_gt, error_rate, indel_rate, erroneous)
                                                    if command != "":
                                                        commands.append((command, infer_net_out_addr, gt_folder))
                        print(scenario, sim, num_gt, method)

    print(f"started, {methods}")
    run_parallel(commands, run_infer_net, max_parallel_processes=cpu_cores,  batch_size=1)
    print(f"Species {num_species} is finished")


def produce_commands_infer_net(phylonet, gt_folder, method_folder, num_gt, iqtree_flag, sites, max_reticulation, method, params_mcmc_gt, error_rate, indel_rate, erroneous):
    iqtree_folder = gt_folder + f"iqtrees/"
    if iqtree_flag:
        gt_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_rooted"
    else:
        gt_addr = f"{gt_folder}genetrees_scaled.txt"
    with open(gt_addr, "r") as h:
        gene_trees = h.read()
    gene_trees = gene_trees.strip().split("\n")
    assert num_gt == len(gene_trees)
    gene_trees = [f"Tree gt{i} = " + gt for i, gt in enumerate(gene_trees)]
    gene_trees = '\n'.join(gene_trees)
    if method == 'InferNetwork_MPL':
        if iqtree_flag:
            infer_net_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.nex"
            infer_net_out_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nInferNetwork_MPL (gt0-gt{num_gt - 1}) {max_reticulation} -di -pl 1;  \nEND;\n "
        else:
            infer_net_addr = f"{method_folder}reticulation_{max_reticulation}.nex"
            infer_net_out_addr = f"{method_folder}reticulation_{max_reticulation}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nInferNetwork_MPL (gt0-gt{num_gt - 1}) {max_reticulation} -di -pl 1;  \nEND;\n "
    elif method == 'InferNetwork_ML':
        if iqtree_flag:
            infer_net_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.nex"
            infer_net_out_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nInferNetwork_ML (gt0-gt{num_gt - 1}) {max_reticulation} -di -pl 1;  \nEND;\n "
        else:
            infer_net_addr = f"{method_folder}reticulation_{max_reticulation}.nex"
            infer_net_out_addr = f"{method_folder}reticulation_{max_reticulation}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nInferNetwork_ML (gt0-gt{num_gt - 1}) {max_reticulation} -di -pl 1;  \nEND;\n "
    elif method == 'MCMC_GT':
        cl = params_mcmc_gt["cl"]
        bl = params_mcmc_gt["bl"]
        sf = params_mcmc_gt["sf"]
        if iqtree_flag:
            infer_net_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.nex"
            infer_net_out_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nMCMC_GT (gt0-gt{num_gt-1}) -cl {cl} -bl {bl} -sf {sf} -mr {max_reticulation} -pl 1 ;  \nEND;\n "
        else:
            infer_net_addr = f"{method_folder}reticulation_{max_reticulation}.nex"
            infer_net_out_addr = f"{method_folder}reticulation_{max_reticulation}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nMCMC_GT (gt0-gt{num_gt - 1}) -cl {cl} -bl {bl} -sf {sf} -mr {max_reticulation} -pl 1 ;  \nEND;\n "

    elif method == 'MCMC_GT_pseudo':
        cl = params_mcmc_gt["cl"]
        bl = params_mcmc_gt["bl"]
        sf = params_mcmc_gt["sf"]
        if iqtree_flag:
            infer_net_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}.nex"
            infer_net_out_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nMCMC_GT (gt0-gt{num_gt - 1}) -cl {cl} -bl {bl} -sf {sf} -mr {max_reticulation} -pl 1 -pseudo ;  \nEND;\n "
        else:
            infer_net_addr = f"{method_folder}reticulation_{max_reticulation}.nex"
            infer_net_out_addr = f"{method_folder}reticulation_{max_reticulation}_out.txt"
            nex_file = f"#NEXUS\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nMCMC_GT (gt0-gt{num_gt - 1}) -cl {cl} -bl {bl} -sf {sf} -mr {max_reticulation} -pl 1 -pseudo ;  \nEND;\n "
    if os.path.isfile(infer_net_out_addr):
        return "", ""
    with open(infer_net_addr, "w") as h:
        h.write(nex_file)
    command = f"/usr/bin/time -v java -jar {phylonet} {infer_net_addr}"
    return command, infer_net_out_addr


def calc_variables(infer_net_out_addr, iqtree_time, mafft_time, sites, scenario, num_species, sim, max_reticulation, num_gt, method, error_rate, indel_rate, mafft_flag, data, iqtree_flag, normalized_distance_truth_vs_mafft_false, normalized_distance_truth_vs_mafft_true, normalized_distance_mafft_true_vs_mafft_false, mafft_sp_scores, mafft_tc_scores, ground_truth_network_likelihood, likelihoods, num_topologies, mismatches, bootstrap_true_vs_original_refinement, bootstrap_true_vs_estimated_refinement, kl_true_vs_original, kl_true_vs_estimated, l1_true_vs_original, l1_true_vs_estimated):
    if os.path.isfile(infer_net_out_addr):
        with open(infer_net_out_addr, "r") as h:
            res = h.read()
        res = res.strip().split("\n")
        temp = {}
        for idx, elm in enumerate(res):
            elm = elm.strip()
            if method == 'MCMC_GT' or method == 'MCMC_GT_pseudo':
                if elm.startswith("Percentage:"):
                    temp["percentage"] = float(elm.split(":")[1].strip())
            elif method == 'InferNetwork_MPL' or method == 'InferNetwork_ML':
                temp["percentage"] = ""
            if elm.startswith("distance_RF:"):
                string_temp = elm.split(":")[1].strip()
                temp["distance_RF"] = float(string_temp) if string_temp != "" else string_temp
            if elm.startswith("distance_luay"):
                string_temp = elm.split(":")[1].strip()
                temp["distance_luay"] = float(string_temp) if string_temp != "" else string_temp
            if elm.startswith("distance_rnbs:"):
                string_temp = elm.split(":")[1].strip()
                temp["distance_rnbs"] = float(string_temp) if string_temp != "" else string_temp
            if elm.startswith("distance_normwapd:"):
                string_temp = elm.split(":")[1].strip()
                temp["distance_normwapd"] = float(string_temp) if string_temp != "" else string_temp
            if elm.startswith("num_inferred_reticulations:"):
                temp["num_inferred_reticulations"] = float(elm.split(":")[1].strip())
            if elm.startswith("num_real_reticulations:"):
                temp["num_real_reticulations"] = float(elm.split(":")[1].strip())
            if elm.startswith("Log Probability:"):
                temp["log_probability"] = float(elm.split(":")[1].strip())
            if elm.startswith("Estimated_network:"):
                temp["estimated_network"] = elm.split(" ")[1].strip()
            if elm.startswith("Target_network:"):
                temp1 = elm.split(" ")
                if len(temp1) == 2:
                    temp["target_network"] = elm.split(" ")[1].strip()
                elif len(temp1) == 3:
                    temp["target_network"] = elm.split("\t")[1].strip().replace(" ", "")
            if elm.strip().startswith("User time (seconds):"):
                user_time = float(elm.split(":")[1].strip())
            if elm.strip().startswith("System time (seconds):"):
                system_time = float(elm.split(":")[1].strip())
        temp["time_infer_net"] = user_time + system_time
        temp["scenario"] = scenario
        temp["alignment_length"] = str(sites)
        temp["num_species"] = num_species
        temp["replica"] = sim
        temp["likelihoods"] = likelihoods
        temp["num_topologies"] = num_topologies
        temp["ground_truth_network_likelihood"] = ground_truth_network_likelihood
        temp["mismatches"] = mismatches
        temp["max_reticulation"] = max_reticulation
        temp["error_rate"] = error_rate
        temp["num_gt"] = num_gt
        temp["iqtree"] = str(iqtree_flag)
        temp["iqtree_time"] = iqtree_time
        temp["method"] = method
        temp["indel_rate"] = indel_rate
        temp["mafft"] = str(mafft_flag)
        temp["mafft_time"] = mafft_time
        temp["norm_avg_dis_truth_vs_mafft_false"] = normalized_distance_truth_vs_mafft_false
        temp["norm_avg_dis_truth_vs_mafft_true"] = normalized_distance_truth_vs_mafft_true
        temp["norm_avg_dis_mafft_true_vs_mafft_false"] = normalized_distance_mafft_true_vs_mafft_false
        temp["mafft_avg_sp_scores"] =mafft_sp_scores
        temp["mafft_avg_tc_scores"] = mafft_tc_scores
        temp["bootstrap_true_vs_original_refinement"] = bootstrap_true_vs_original_refinement
        temp["bootstrap_true_vs_estimated_refinement"] = bootstrap_true_vs_estimated_refinement
        temp["kl_true_vs_original"] = kl_true_vs_original
        temp["kl_true_vs_estimated"] = kl_true_vs_estimated
        temp["l1_true_vs_original"] = l1_true_vs_original
        temp["l1_true_vs_estimated"] = l1_true_vs_estimated

    else:
        print(f"not available: {infer_net_out_addr}")
        temp = {}
        temp["log_probability"] = temp["estimated_network"] = temp["target_network"] = temp["time_infer_net"] = temp[
            "iqtree_time"] = temp["scenario"] = \
            temp["iqtree"] = temp["alignment_length"] = temp["num_species"] = temp["max_reticulation"] = temp[
            "num_gt"] = temp["distance_RF"] = \
            temp["distance_luay"] = temp["distance_rnbs"] = temp["distance_normwapd"] = temp["num_real_reticulations"] = \
        temp["num_inferred_reticulations"] = temp["mafft_time"] = temp["error_rate"] =  temp["method"]= temp["indel_rate"]= temp["mafft"] = temp["percentage"] = temp["replica"] =\
            temp["norm_avg_dis_truth_vs_mafft_false"] = temp["norm_avg_dis_truth_vs_mafft_true"] = temp["norm_avg_dis_mafft_true_vs_mafft_false"] = temp["mafft_avg_sp_scores"] = \
            temp["mafft_avg_tc_scores"] =  temp["ground_truth_network_likelihood"] = temp["likelihoods"] = temp["num_topologies"] =temp["mismatches"]= \
            temp["bootstrap_true_vs_original_refinement"] = temp["bootstrap_true_vs_estimated_refinement"] = \
            temp["kl_true_vs_original"] = temp["kl_true_vs_estimated"] = temp["l1_true_vs_original"] = temp["l1_true_vs_estimated"] = ""
    data.append(temp)
    return data


def calc_likelihood(infer_net_out_addr, method):
    if os.path.isfile(infer_net_out_addr):
        with open(infer_net_out_addr, "r") as h:
            res = h.read()
        res = res.strip().split("\n")
        temp = {}
        if method in ['InferNetwork_MPL', 'InferNetwork_ML']:
            num_topologies = 1
            for idx, elm in enumerate(res):
                if elm.startswith("Results after run"):
                    run_number = int(elm.split("#")[1])
                    temp[run_number] = []
                elif elm.startswith("-"):
                    temp[run_number].append(float(elm.split(":")[0]))
                elif elm.startswith("Total log probability:"):

                    if "Running time:" in elm:
                        best_likelihood = float(elm.split(":")[1].strip().split("Running time")[0].strip())
                    else:
                        best_likelihood = float(elm.split(":")[1].strip())
                    log_liklihood_lst = [val[0] for _, val in temp.items()]
                    # assert max(log_liklihood_lst) <= round(best_likelihood, 4), print(infer_net_out_addr)   #TODO check this
            val_str = ""
            for idx in range(1, max(temp.keys()) + 1):
                val_str += str(temp[idx]) + ";"
            val_str = val_str[:-1]
        elif method=='MCMC_GT_pseudo':
            num_topologies = 0
            for idx, elm in enumerate(res):
                line = elm.strip().split(";")
                if len(line)==8 and line[0][0] >= "0" and line[0][0] <= "9":
                    temp2 = {}
                    iter = int(line[0].strip())
                    if iter != 0:
                        temp2["Iteration"] = float(line[0].strip())
                        temp2["Posterior"] = float(line[1].strip())
                        temp2["ESS"] = float(line[2].strip())
                        temp2["Likelihood"] = float(line[3].strip())
                        temp2["Prior"] = float(line[4].strip())
                        temp2["ESS_prior"] = float(line[5].strip())
                        temp2["Reticulation"] = float(line[6].strip())
                        temp[iter] = temp2

                elif elm.startswith("Rank ="):
                    num_topologies += 1
                    best_likelihood = float(line[3].strip().split("=")[1].strip().split(":")[0].strip())
                    log_liklihood_lst = [val["Posterior"] for _, val in temp.items()]

                    # assert max([val["Posterior"] for _, val in temp.items()]) <= round(best_likelihood, 4) , print(infer_net_out_addr)   #TODO check this

            assert 1 in temp.keys() and 1000 in temp.keys()
            val_str = ""
            for idx in range(1, max(temp.keys()) + 1):
                val_str += str([temp[idx]["Posterior"]]) + ";"
            val_str = val_str[:-1]
        else:
            raise Exception("method is wrong")


    else:
        print(f"not available: {infer_net_out_addr}")
        val_str = ""

    return val_str, num_topologies

def calc_sum_times(addr):
    if os.path.isfile(addr):
        with open(addr, "r") as h:
            lst_times = h.read()
        sum_time = sum(
            [float(item) for item in lst_times.strip().split("\n")])
    else:
        raise Exception(f"why this file is not available: {addr}")
    return sum_time

def calc_average_items(addr):
    if os.path.isfile(addr):
        with open(addr, "r") as h:
            lst_times = h.read()
        average = np.mean([float(item) for item in lst_times.strip().split("\n")])
    else:
        raise Exception(f"why this file is not available: {addr}")
    return average

def get_num_mismatches(gt_folder, sites, indel_rate, error_rate):
    if error_rate == 0:
        mismatch_addr = f"{gt_folder}alignments/mismatch_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_original.txt"
        with open(mismatch_addr, "r") as h:
            mismatches = float(h.read().strip())
    else:
        mismatches = ""
    return mismatches


def calc_kl_l1(addr):
    if os.path.isfile(addr):
        with open(addr, "r") as h:
            kl_l1 = h.read()
    kl_lst = kl_l1.splitlines()[0].split('\t')
    kl_lst = [i.split(":") for i in kl_lst]
    kl_dic = {key.strip():float(val) for key, val in kl_lst}
    kl_true_vs_original = kl_dic['true_vs_original_kl']
    kl_true_vs_estimated = kl_dic['true_vs_estimated_kl']
    kl_original_vs_estimated = kl_dic['original_vs_estimated_kl']

    l1_lst = kl_l1.splitlines()[1].split('\t')
    l1_lst = [i.split(":") for i in l1_lst]
    l1_dic = {key.strip():float(val) for key, val in l1_lst}
    l1_true_vs_original = l1_dic['true_vs_original_l1']
    l1_true_vs_estimated = l1_dic['true_vs_estimated_l1']
    l1_original_vs_estimated = l1_dic['original_vs_estimated_l1']

    return  kl_true_vs_original, kl_true_vs_estimated, kl_original_vs_estimated, l1_true_vs_original, l1_true_vs_estimated, l1_original_vs_estimated

def create_output(scenarios, root_folder, num_tip_lst, numbsim, start_replica, num_gt_lst, sites_per_gt_lst, lst_max_reticulation, error_rate_lst, INDEL_RATE_LST, methods):
    data = []
    results_folder = f"{root_folder}results/"
    os.makedirs(results_folder, exist_ok=True)
    csv_addr = f"{results_folder}final_result.csv"
    for scenario in scenarios:
        scenario_folder = f"{root_folder}{scenario}/"
        for i, num_species in enumerate(num_tip_lst):
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    iqtree_folder = gt_folder + f"iqtrees/"
                    mafft_folder = gt_folder + f"maffts/"
                    for method in methods:
                        method_folder = gt_folder +  f"{method}/"
                        for sites in sites_per_gt_lst:
                            for iqtree_flag in [False, True]:
                                for max_reticulation in lst_max_reticulation:
                                    if iqtree_flag:
                                        for error_rate in error_rate_lst:
                                            for indel_rate in INDEL_RATE_LST:
                                                for erroneous in ["original", "estimated_mafft"]:
                                                    mafft_flag = True if erroneous=="estimated_mafft" else False

                                                    iqtree_time_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_times"
                                                    iqtree_time = calc_sum_times(iqtree_time_addr)
                                                    iqtree_gt_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_rooted"

                                                    if mafft_flag == False :
                                                        true_vs_original_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_true_vs_original"
                                                        normalized_distance_truth_vs_mafft_false = calc_average_items(true_vs_original_addr)
                                                        normalized_distance_truth_vs_mafft_true = ""
                                                        normalized_distance_mafft_true_vs_mafft_false = ""
                                                        mafft_time = ""
                                                        mafft_sp_scores = ""
                                                        mafft_tc_scores = ""

                                                        bootstrap_true_vs_original_refinement, bootstrap_true_vs_estimated_refinement, _ = calc_refinement_gene_trees(gt_folder, num_gt, sites, iqtree_folder, indel_rate, error_rate)
                                                        bootstrap_true_vs_estimated_refinement = ""

                                                        kl_file_addr = iqtree_folder + f"distance_kl_l1_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}.txt"
                                                        kl_true_vs_original, kl_true_vs_estimated, _, l1_true_vs_original, l1_true_vs_estimated, _ = calc_kl_l1(kl_file_addr)
                                                        kl_true_vs_estimated = ""
                                                        l1_true_vs_estimated = ""


                                                    elif mafft_flag == True:
                                                        truth_vs_mafft_false_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_true_vs_original"
                                                        truth_vs_mafft_true_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_true_vs_estimated"
                                                        mafft_true_vs_mafft_false_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_original_vs_estimated"
                                                        normalized_distance_truth_vs_mafft_false = calc_average_items(truth_vs_mafft_false_addr)
                                                        normalized_distance_truth_vs_mafft_true = calc_average_items(truth_vs_mafft_true_addr)
                                                        normalized_distance_mafft_true_vs_mafft_false = calc_average_items(mafft_true_vs_mafft_false_addr)
                                                        mafft_times_addr = mafft_folder + f"mafft_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_times"
                                                        mafft_time = calc_sum_times(mafft_times_addr)
                                                        mafft_sp_scores_addr = mafft_folder + f"mafft_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_sp_scores"
                                                        mafft_tc_scores_addr = mafft_folder + f"mafft_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_tc_scores"
                                                        mafft_sp_scores = calc_average_items(mafft_sp_scores_addr)
                                                        mafft_tc_scores = calc_average_items(mafft_tc_scores_addr)

                                                        bootstrap_true_vs_original_refinement, bootstrap_true_vs_estimated_refinement, _ = calc_refinement_gene_trees(gt_folder, num_gt, sites, iqtree_folder, indel_rate, error_rate)
                                                        bootstrap_true_vs_original_refinement = ""

                                                        kl_file_addr = iqtree_folder + f"distance_kl_l1_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}.txt"
                                                        kl_true_vs_original, kl_true_vs_estimated, _, l1_true_vs_original, l1_true_vs_estimated, _ = calc_kl_l1(kl_file_addr)
                                                        kl_true_vs_original = ""
                                                        l1_true_vs_original = ""

                                                    mismatches = get_num_mismatches(gt_folder, sites, indel_rate, error_rate)
                                                    infer_net_out_addr = f"{method_folder}iqtree_reticulation_{max_reticulation}_length_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_{erroneous}_out.txt"
                                                    ground_truth_network_likelihood = ""
                                                    likelihoods, num_topologies = calc_likelihood(infer_net_out_addr, method)
                                                    data = calc_variables(infer_net_out_addr, iqtree_time, mafft_time, sites,
                                                                          scenario, num_species, sim, max_reticulation, num_gt, method,
                                                                          error_rate, indel_rate, mafft_flag, data, iqtree_flag, normalized_distance_truth_vs_mafft_false, normalized_distance_truth_vs_mafft_true,
                                                              normalized_distance_mafft_true_vs_mafft_false, mafft_sp_scores, mafft_tc_scores, ground_truth_network_likelihood, likelihoods, num_topologies, mismatches,
                                                              bootstrap_true_vs_original_refinement, bootstrap_true_vs_estimated_refinement,
                                                              kl_true_vs_original, kl_true_vs_estimated, l1_true_vs_original, l1_true_vs_estimated)
                                    else:
                                        error_rate = 0
                                        indel_rate = 0
                                        iqtree_time = ""
                                        normalized_distance_truth_vs_mafft_false = ""
                                        normalized_distance_truth_vs_mafft_true = ""
                                        normalized_distance_mafft_true_vs_mafft_false = ""
                                        mafft_time = ""
                                        mafft_sp_scores = ""
                                        mafft_tc_scores = ""
                                        mafft_flag = False
                                        bootstrap_true_vs_original_refinement = ""
                                        bootstrap_true_vs_estimated_refinement = ""
                                        kl_true_vs_original = ""
                                        kl_true_vs_estimated = ""
                                        l1_true_vs_original = ""
                                        l1_true_vs_estimated = ""

                                        mismatches = get_num_mismatches(gt_folder, sites, indel_rate, error_rate)
                                        infer_net_out_addr = f"{method_folder}reticulation_{max_reticulation}_out.txt"
                                        ground_truth_network_likelihood = calc_ground_truth_network_likelihood(gt_folder, method_folder, max_reticulation, num_gt, phylonet, method)
                                        likelihoods, num_topologies = calc_likelihood(infer_net_out_addr, method)
                                        data = calc_variables(infer_net_out_addr, iqtree_time, mafft_time, sites, scenario,
                                                              num_species, sim, max_reticulation, num_gt, method, error_rate, indel_rate, mafft_flag,
                                                              data, iqtree_flag, normalized_distance_truth_vs_mafft_false, normalized_distance_truth_vs_mafft_true,
                                                              normalized_distance_mafft_true_vs_mafft_false, mafft_sp_scores, mafft_tc_scores, ground_truth_network_likelihood, likelihoods, num_topologies, mismatches,
                                                              bootstrap_true_vs_original_refinement, bootstrap_true_vs_estimated_refinement,
                                                              kl_true_vs_original, kl_true_vs_estimated, l1_true_vs_original, l1_true_vs_estimated)
                                    print(f"scenario:{scenario} sim:{sim} num_gt:{num_gt} method:{method} sites:{sites} iqtree_flag:{iqtree_flag} max_reticulation:{max_reticulation}")

        print(f"Scenario: {scenario} is finished")
    for row in data:
        if "iqtree" not in row.keys():
            print()
    with open(csv_addr, 'w', newline='') as csvfile:
        fieldnames = ['scenario', 'method', 'num_species', 'replica', 'num_gt', 'alignment_length', 'indel_rate', 'error_rate', 'max_reticulation', 'mafft',
                      'iqtree', 'num_real_reticulations', 'num_inferred_reticulations', 'distance_RF',
                      'distance_luay', 'distance_rnbs', 'distance_normwapd', 'percentage', 'ground_truth_network_likelihood', 'log_probability', 'time_infer_net',
                      'iqtree_time', "mafft_time", 'norm_avg_dis_truth_vs_mafft_false', 'norm_avg_dis_truth_vs_mafft_true', 'norm_avg_dis_mafft_true_vs_mafft_false',
                      'mafft_avg_sp_scores', 'mafft_avg_tc_scores', 'num_topologies', 'mismatches', 'bootstrap_true_vs_original_refinement', 'bootstrap_true_vs_estimated_refinement',
                                                              'kl_true_vs_original', 'kl_true_vs_estimated', 'l1_true_vs_original', 'l1_true_vs_estimated','estimated_network', 'target_network', 'likelihoods']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def prepare_network_for_distance(net):
    temp = net.split("::")
    if len(temp) != 1:
        for i in temp[1:]:
            remove_val = "::" + i.split(",")[0].split(")")[0]
            net = net.replace(remove_val, "")
    return net

def  calc_ground_truth_network_likelihood(gt_folder, method_folder, max_reticulation, num_gt, phylonet, method):
    ground_truth_net_addr = f"{gt_folder}species.nw"
    with open(ground_truth_net_addr, "r") as h:
        ground_truth_net = h.read()
    gt_addr = f"{gt_folder}genetrees_scaled.txt"
    with open(gt_addr, "r") as h:
        gene_trees = h.read()
    gene_trees = gene_trees.strip().split("\n")
    gene_trees = [f"Tree gt{i} = " + gt for i, gt in enumerate(gene_trees)]
    gene_trees = '\n'.join(gene_trees)
    likelihood_net_nex_addr = f"{method_folder}reticulation_{max_reticulation}_likelihood.nex"
    likelihood_net_out_addr = f"{method_folder}reticulation_{max_reticulation}_likelihood_out.txt"
    if method== "MCMC_GT_pseudo":
        nex_file = f"#NEXUS\n  BEGIN NETWORKS; \n Network net = {ground_truth_net} \n  END;\n\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nCalGTProb net (gt0-gt{num_gt - 1}) -pseudo -pl 1;  \nEND;\n "
    else:
        nex_file = f"#NEXUS\n  BEGIN NETWORKS; \n Network net = {ground_truth_net} \n  END;\n\n BEGIN TREES;\n{gene_trees}\nEND;\n" + f"\nBEGIN PHYLONET;\nCalGTProb net (gt0-gt{num_gt - 1}) -pl 1;  \nEND;\n "
    with open(likelihood_net_nex_addr, "w") as h:
        h.write(nex_file)
    command = f"java -jar {phylonet} {likelihood_net_nex_addr}"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    stdout, stderr = process.communicate()
    if stderr!="":
        print("")
    likelihood_net = float(stdout.strip().split("\n")[3].split(":")[1].strip())
    return str(likelihood_net)


def  calc_net_distance(estimated_network, Target_network, results_folder):
    T1 =copy.deepcopy(Target_network)
    estimated_network = prepare_network_for_distance(estimated_network)
    Target_network = prepare_network_for_distance(Target_network)
    import uuid

    # Generate a random UUID
    unique_id = str(uuid.uuid4())
    infer_net_distance_addr = f"{results_folder}distance_{unique_id}.nex"
    distance_luay = ""
    distance_rnbs = ""
    distance_normwapd = ""
    for Characterization_mode in ["luay", "rnbs", "normwapd"]:
        nex_file = f"#NEXUS\n BEGIN NETWORKS;\n Network net1 = {Target_network} \n Network net2 = {estimated_network} \nEND;\n" + f"\nBEGIN PHYLONET;\nCmpnets net1 net2 -m {Characterization_mode}; \nEND;\n "
        with open(infer_net_distance_addr, "w") as h:
            h.write(nex_file)
        # time.sleep(1)
        command = f"java -jar {phylonet} {infer_net_distance_addr}"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        stdout, stderr = process.communicate()
        if stderr!="":
            print()
        temp = stdout.strip().split("\n")[1]
        if temp.startswith("The Luay's distance between two networks:"):
            distance_luay = float(temp.split(":")[1].strip())
        elif temp.startswith("rNBS dissimilarity between two networks:"):
            distance_rnbs = float(temp.split(":")[1].strip())
        elif temp.startswith("NormWAPD dissimilarity between two networks:"):
            distance_normwapd = float(temp.split(":")[1].strip())
    os.remove(infer_net_distance_addr)
    return distance_luay, distance_rnbs, distance_normwapd

def calc_RF_distance(tree1, tree2):
    # Create a shared taxon namespace
    shared_taxa = dendropy.TaxonNamespace()
    t1 = dendropy.Tree.get(data=tree1, schema="newick", taxon_namespace=shared_taxa, rooting="default-rooted")
    t2 = dendropy.Tree.get(data=tree2, schema="newick", taxon_namespace=shared_taxa, rooting="default-rooted")
    return treecompare.symmetric_difference(t1, t2)


def calc_RF_distance_gene_trees(scenarios, root_folder, error_rate_lst, INDEL_RATE_LST, numbsim, start_replica):
    for scenario in scenarios:
        scenario_folder = f"{root_folder}{scenario}/"
        for i, num_species in enumerate(num_tip_lst):
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    iqtree_folder = gt_folder + f"iqtrees/"
                    for sites in sites_per_gt_lst:
                        for indel_rate in INDEL_RATE_LST:
                            for error_rate in error_rate_lst:
                                true_vs_original_lst = []
                                true_vs_estimated_lst = []
                                original_vs_estimated_lst = []
                                true_gt_addr = gt_folder + "genetrees.txt"
                                true_gt_addr = gt_folder + "genetrees_scaled.txt"

                                original_iqtree_rooted_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_original_rooted"
                                estimated_mafft_iqtree_rooted_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_estimated_mafft_rooted"
                                with open(true_gt_addr, "r") as f:
                                    true_gt = f.read().strip().split("\n")
                                with open(original_iqtree_rooted_addr, "r") as f:
                                    original_iqtree_rooted = f.read().strip().split("\n")
                                with open(estimated_mafft_iqtree_rooted_addr, "r") as f:
                                    estimated_mafft_iqtree_rooted = f.read().strip().split("\n")
                                    shared_taxa = dendropy.TaxonNamespace()
                                for j in range(num_gt):
                                        true_tree_0 = dendropy.Tree.get(data=true_gt[j], schema="newick",taxon_namespace=shared_taxa, rooting="default-rooted")
                                        original_tree_1 = dendropy.Tree.get(data=original_iqtree_rooted[j], schema="newick", taxon_namespace=shared_taxa, rooting="default-rooted")
                                        estimated_tree_2 = dendropy.Tree.get(data=estimated_mafft_iqtree_rooted[j], schema="newick", taxon_namespace=shared_taxa, rooting="default-rooted")
                                        species_lst = [taxon.label for taxon in true_tree_0.taxon_namespace]
                                        max_rf_distance = (2 * len(species_lst)) - 2
                                        true_vs_original = treecompare.symmetric_difference(true_tree_0, original_tree_1) / max_rf_distance
                                        true_vs_estimated = treecompare.symmetric_difference(true_tree_0, estimated_tree_2) / max_rf_distance
                                        original_vs_estimated = treecompare.symmetric_difference(original_tree_1, estimated_tree_2) / max_rf_distance
                                        true_vs_original_lst.append(true_vs_original)
                                        true_vs_estimated_lst.append(true_vs_estimated)
                                        original_vs_estimated_lst.append(original_vs_estimated)
                                true_vs_original_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_true_vs_original"
                                true_vs_estimated_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_true_vs_estimated"
                                original_vs_estimated_addr = iqtree_folder + f"distance_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_original_vs_estimated"
                                with open(true_vs_original_addr, "w") as h:
                                    h.write("\n".join(map(str, true_vs_original_lst)))
                                with open(true_vs_estimated_addr, "w") as h:
                                    h.write("\n".join(map(str, true_vs_estimated_lst)))
                                with open(original_vs_estimated_addr, "w") as h:
                                    h.write("\n".join(map(str, original_vs_estimated_lst)))
                print(f"scenario:{scenario}  sim:{sim}")


def calc_kl_divergence_l1_distance_gene_trees(scenarios, root_folder, error_rate_lst, INDEL_RATE_LST, numbsim, start_replica):

    def load_topology_ids(file_path):
        """
        Load a list of topology IDs from Newick trees in a file.
        """
        with open(file_path) as f:
            lines = f.read().strip().splitlines()

        topo_ids = []
        for line in lines:
            tree = Tree(line, format=1)
            tree.unroot(mode='keep')
            topo_ids.append(tree.get_topology_id())
        return topo_ids

    def get_distribution(topology_ids):
        """
        Turn a list of topology IDs into a normalized frequency distribution.
        """
        counts = Counter(topology_ids)
        total = sum(counts.values())
        return {topo: count / total for topo, count in counts.items()}


    def kl_divergence(p, q, epsilon=1e-5):
        """
        Compute KL divergence from p to q: D_KL(p || q)
        Add smoothing epsilon to avoid log(0).
        """
        temp = []
        all_keys = set(p) | set(q)
        for k in all_keys:
            temp.append((p.get(k, 0)+ epsilon) * math.log((p.get(k, 0) + epsilon) / (q.get(k, 0) + epsilon)))
        return sum(temp)

    def l1_distance(p, q):
        """
        Compute the L1 distance between two probability distributions p and q.
        p and q should be dictionaries: {topology_id: probability}.
        """
        all_keys = set(p) | set(q)  # Union of all keys
        return sum(abs(p.get(k, 0) - q.get(k, 0)) for k in all_keys)


    for scenario in scenarios:
        scenario_folder = f"{root_folder}{scenario}/"
        for i, num_species in enumerate(num_tip_lst):
            for sim in range(start_replica, numbsim):
                sim_folder = scenario_folder + f"net_{num_species}_species/{sim}/"
                for num_gt in num_gt_lst:
                    gt_folder = sim_folder + f"{num_gt}_gene_trees/"
                    iqtree_folder = gt_folder + f"iqtrees/"
                    for sites in sites_per_gt_lst:
                        for indel_rate in INDEL_RATE_LST:
                            for error_rate in error_rate_lst:
                                true_gt_addr = gt_folder + "genetrees.txt"
                                true_gt_addr = gt_folder + "genetrees_scaled.txt"
                                original_iqtree_rooted_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_original_rooted"
                                estimated_mafft_iqtree_rooted_addr = iqtree_folder + f"iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_estimated_mafft_rooted"

                                true_gt =load_topology_ids(true_gt_addr)
                                original_iqtree_rooted = load_topology_ids(original_iqtree_rooted_addr)
                                estimated_mafft_iqtree_rooted = load_topology_ids(estimated_mafft_iqtree_rooted_addr)

                                true_gt_dist = get_distribution(true_gt)
                                original_iqtree_rooted_dist = get_distribution(original_iqtree_rooted)
                                estimated_mafft_iqtree_rooted_dist = get_distribution(estimated_mafft_iqtree_rooted)

                                true_vs_original_kl = kl_divergence(true_gt_dist, original_iqtree_rooted_dist)
                                true_vs_estimated_kl = kl_divergence(true_gt_dist, estimated_mafft_iqtree_rooted_dist)
                                original_vs_estimated_kl = kl_divergence(original_iqtree_rooted_dist, estimated_mafft_iqtree_rooted_dist)

                                out_str = f"true_vs_original_kl:{true_vs_original_kl} \t true_vs_estimated_kl:{true_vs_estimated_kl} \t original_vs_estimated_kl:{original_vs_estimated_kl}"

                                true_vs_original_l1 = l1_distance(true_gt_dist, original_iqtree_rooted_dist)
                                true_vs_estimated_l1 = l1_distance(true_gt_dist, estimated_mafft_iqtree_rooted_dist)
                                original_vs_estimated_l1 = l1_distance(original_iqtree_rooted_dist, estimated_mafft_iqtree_rooted_dist)

                                out_str += f"\ntrue_vs_original_l1:{true_vs_original_l1} \t true_vs_estimated_l1:{true_vs_estimated_l1} \t original_vs_estimated_l1:{original_vs_estimated_l1}"

                                kl_file_addr = iqtree_folder + f"distance_kl_l1_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}.txt"
                                with open(kl_file_addr, "w") as h:
                                    h.write(out_str)
                print(f"scenario:{scenario}  sim:{sim}")


def calc_refinement_gene_trees(gt_folder, num_gt, sites, iqtree_folder, indel_rate, error_rate):
    def load_topology(file_path):
        with open(file_path) as f:
            lines = f.read().strip().splitlines()

        topo = []
        for line in lines:
            tree = Tree(line, format=1)
            tree.unroot(mode='keep')
            topo.append(tree)
        return topo

    def get_splits(tree):
        """
        Get all bipartitions (splits) from a tree as sets of leaf names.
        """
        splits = set()
        leaves = set(tree.get_leaf_names())
        for node in tree.traverse():
            if not node.is_leaf():
                node_leaves = set(node.get_leaf_names())
                if 0 < len(node_leaves) < len(leaves):
                    splits.add(frozenset(node_leaves))
        return splits

    def is_refinement(true_splits, contracted_splits):
        return 1 if contracted_splits.issubset(true_splits) else 0

    true_vs_original_lst = []
    true_vs_estimated_lst = []
    original_vs_estimated_lst = []
    true_gt_addr = gt_folder + "genetrees.txt"
    true_gt_addr = gt_folder + "genetrees_scaled.txt"
    original_bootstrap_iqtree_rooted_addr = iqtree_folder + f"bootstrap_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_original_rooted"
    estimated_mafft_bootstrap_iqtree_rooted_addr = iqtree_folder + f"bootstrap_iqtree_{sites}_indel_rate_{indel_rate}_error_{error_rate}_alignment_estimated_mafft_rooted"
    true_gt = load_topology(true_gt_addr)
    original_bootstrap_iqtree_rooted = load_topology(original_bootstrap_iqtree_rooted_addr)
    estimated_mafft_bootstrap_iqtree_rooted = load_topology(estimated_mafft_bootstrap_iqtree_rooted_addr)

    for j in range(num_gt):
        true_gt_splits = get_splits(true_gt[j])
        original_bootstrap_iqtree_rooted_splits = get_splits(original_bootstrap_iqtree_rooted[j])
        estimated_mafft_bootstrap_iqtree_rooted_splits = get_splits(estimated_mafft_bootstrap_iqtree_rooted[j])
        true_vs_original = is_refinement(true_gt_splits, original_bootstrap_iqtree_rooted_splits)
        true_vs_estimated = is_refinement(true_gt_splits, estimated_mafft_bootstrap_iqtree_rooted_splits)
        original_vs_estimated = is_refinement(original_bootstrap_iqtree_rooted_splits, estimated_mafft_bootstrap_iqtree_rooted_splits)

        true_vs_original_lst.append(true_vs_original)
        true_vs_estimated_lst.append(true_vs_estimated)
        original_vs_estimated_lst.append(original_vs_estimated)
    assert len(true_vs_original_lst) == num_gt
    return sum(true_vs_original_lst) / num_gt, sum(true_vs_estimated_lst) / num_gt, sum(original_vs_estimated_lst) / num_gt




def read_result_file(results_folder):

    csv_addr = f"{results_folder}final_result.csv"
    data = pd.read_csv(csv_addr)
    mask = ~pd.isna(data['scenario'])
    data = data[mask]
    data["num_species"] = data["num_species"].astype(int)
    data["num_gt"] = data["num_gt"].astype(int)
    data["replica"] = data["replica"].astype(int)
    data["alignment_length"] = data["alignment_length"].astype(int)
    data["ground_truth_network_likelihood"] = data["ground_truth_network_likelihood"].astype(float)
    data["error_rate"] = data["error_rate"].astype(float)
    data["max_reticulation"] = data["max_reticulation"].astype(int)
    data["mafft"] = data["mafft"].astype(bool)
    data["iqtree"] = data["iqtree"].astype(bool)
    data["indel_rate"] = data["indel_rate"].astype(float)
    data["num_real_reticulations"] = data["num_real_reticulations"].astype(int)
    data["num_inferred_reticulations"] = data["num_inferred_reticulations"].astype(int)
    scenarioes = data['scenario'].unique()
    methods = data['method'].unique()
    num_species = data['num_species'].unique()
    num_gts = data['num_gt'].unique()
    replicas = data['replica'].unique()
    alignment_lengths = data['alignment_length'].unique()
    max_reticulations = data['max_reticulation'].unique()
    maffts = data['mafft'].unique()
    iqtrees = data['iqtree'].unique()
    num_real_reticulations = data['num_real_reticulations'].unique()
    return data, scenarioes, methods, num_species, num_gts, replicas, alignment_lengths, max_reticulations, iqtrees, maffts, num_real_reticulations

def box_plot_generatoin(subset, columns, axe, title, xlabel, ylabel, legend_title):
    custom_palette = ['#f07c24', '#1b86f2', '#32a852', '#ed2d73']
    custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    sub_subset = subset[columns]
    # sub_subset.convert_dtypes()
    # plt.figure(figsize=(8, 8))
    num_color = len(sub_subset[columns[2]].unique())
    custom_palette = custom_palette[0:num_color]
    sns.boxplot(x=columns[0], y=columns[1], hue=columns[2], data=sub_subset.dropna(),
                ax=axe, width=0.4, palette=custom_palette)

    axe.set_title(title)
    axe.set_xlabel(xlabel, fontsize=14)
    axe.set_ylabel(ylabel, fontsize=14)
    axe.legend(title=legend_title, fontsize=11, title_fontsize=12)  # Adjust legend font sizes
    # dic_axes[j].set_xticks(ticks=num_gts,rotation=45, fontsize=12)
    # dic_axes[j].set_yticks(ticks=subset["diff"], fontsize=12)
    return axe

def box_plot_generation_with_palette(subset, columns, axe, title, xlabel, ylabel, legend_title, custom_palette):
    sub_subset = subset[columns]
    num_color = len(sub_subset[columns[2]].unique())
    # custom_palette = custom_palette[0:num_color]
    sns.boxplot(x=columns[0], y=columns[1], hue=columns[2], data=sub_subset.dropna(),
                ax=axe, width=0.4, palette=custom_palette)
    axe.set_title(title)
    axe.set_xlabel(xlabel, fontsize=14)
    axe.set_ylabel(ylabel, fontsize=14)
    axe.legend(title=legend_title, fontsize=11, title_fontsize=12)  # Adjust legend font sizes
    # dic_axes[j].set_xticks(ticks=num_gts,rotation=45, fontsize=12)
    # dic_axes[j].set_yticks(ticks=subset["diff"], fontsize=12)
    return axe

def line_plot_generation_with_palette(subset, columns, axe, title, xlabel, ylabel, legend_title, custom_palette):
    sub_subset = subset[columns]
    num_color = len(sub_subset[columns[2]].unique())
    # custom_palette = custom_palette[0:num_color]
    # sns.boxplot(x=, y=columns[1], hue=columns[2], data=sub_subset.dropna(),
    #             ax=axe, width=0.4, palette=custom_palette)
    sns.lineplot(data=sub_subset.dropna(),
                    x=columns[0], y=columns[1], style=columns[2], hue=columns[2],
                    errorbar=("ci", 50), err_style="band",
                    markers=True, dashes=False, palette="deep",
                    ax=axe)
    axe.set_title(title)
    axe.set_xlabel(xlabel, fontsize=11)
    axe.set_ylabel(ylabel, fontsize=11)
    axe.legend(title=legend_title, fontsize=11, title_fontsize=12)  # Adjust legend font sizes
    # dic_axes[j].set_xticks(ticks=num_gts,rotation=45, fontsize=12)
    # dic_axes[j].set_yticks(ticks=subset["diff"], fontsize=12)
    return axe

def create_box_plot_mafft_scores_for_paper(root_folder, results_folder, data, scenarioes, methods, num_species, num_gts, alignment_lengths, max_reticulations, iqtrees, maffts, num_real_reticulations, replicas, error_rate_lst, INDEL_RATE_LST):
    dic_names = {'scenario': "Scenario", "num_gt": "# Gene Trees", "method": "Method", "alignment_length": "Alignment Length",
     "error_rate": "Error Rate", "indel_rate":"Indel Rate", "iqtree":"IQTREE", "mafft":"MAFFT", "max_reticulation":"Max Reticulation"}
    # lst_parameters = {'scenario':scenarioes, "num_gt":num_gts, "method":methods, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
    #                   "indel_rate":INDEL_RATE_LST, "iqtree":iqtrees, "mafft":maffts, "max_reticulation":lst_max_reticulation}
    lst_parameters = {'scenario':scenarioes, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
                      "indel_rate":INDEL_RATE_LST}
    color_variables = [ "indel_rate"]
    x_axis_variables = ["alignment_length"]
    lst_masks = []
    count_all_figures = 0
    all_two_variables = list(itertools.product(*[x_axis_variables, color_variables]))
    for variables in all_two_variables:
        print(variables)
        fixed_parameter_lst = [i for i in lst_parameters.keys() if i not in variables and i not in [ "error_rate"]]
        # fixed_parameter_lst = [i for i in  if i not in variables]

        assert len(fixed_parameter_lst) == 1
        all_combinations = list(itertools.product(*[lst_parameters[i] for i in fixed_parameter_lst]))
        count_all_figures += len(all_combinations)
        for idx, combo in enumerate(all_combinations):
            # df[cond1 & cond2 & cond3 & cond4 & cond5]
            temp = np.array(list(zip(fixed_parameter_lst+ ["mafft", "method", "max_reticulation"], list(combo) + [True, 'InferNetwork_MPL', np.unique(data["max_reticulation"])[0]])))
            all_params = {j:"Varies" for j in variables }
            for key, value in temp:
                all_params[key] = value
            mask = [1] * len(data)
            for i, item in enumerate(temp):
                if temp[i, 0] == "num_gt" or temp[i, 0] =="alignment_length" or  temp[i, 0] =="max_reticulation":
                    val = int(temp[i, 1])
                elif temp[i, 0] =="error_rate" or temp[i, 0] =="indel_rate":
                    val = float(temp[i, 1])
                elif temp[i, 0] =="iqtree" or temp[i, 0] =="mafft":
                    if temp[i, 1]=="True":
                        val = True
                    else:
                        val=False
                else:
                    val =  temp[i, 1]
                mask = mask & np.array(data[temp[i, 0]] == val)
                # print()
                # data.iloc[mask.astype(bool)]
            # mask = np.array(data['scenario'] == scenario) & np.array(data['num_gt'] == num_gt) & np.array(data['method'] == method)
            lst_masks.append(mask)
            assert sum(lst_masks[0]) == len(replicas) * len(num_gts) * len(alignment_lengths) * len(error_rate_lst) * len(INDEL_RATE_LST)
            suptitle = f'mafft_score__'
            for key in lst_parameters.keys():
                if key not in [ "indel_rate", "error_rate"]:
                    suptitle += f'{key}:{all_params[key]}__'
            suptitle = suptitle[:-2]
            save_addr = f"{results_folder}paper/mafft_score/{all_params['scenario']}/{'--'.join(variables)}/"
            os.makedirs(save_addr, exist_ok=True)
            save_addr +=  f"{suptitle}.png"
            if os.path.isfile(save_addr):
                continue

            suptitle = f"MAFFT Average Scores for Different {dic_names[variables[0]]}"
            subset = data.iloc[mask.astype(bool)]
            num_panels = 0
            available_panels = []
            # for distance in lst_distance:
            #     if len(subset[[variables[0], distance, variables[1]]].dropna()) > 0:
            #         num_panels += 1
            #         available_panels.append(distance)
            num_panels = 3
            # available_panels = ["mafft_avg_sp_scores", "mafft_avg_tc_scores"]
            available_panels = list(np.unique(subset["error_rate"]))
            dic_scores = {"mafft_avg_sp_scores": "MAFFT Average SP Score", "mafft_avg_tc_scores":"MAFFT Average TC Score"}
            if num_panels == 0:
                continue
            else:
                fig, axes = plt.subplots(1, num_panels, figsize=(num_panels * 4, 6), squeeze=False, dpi=300, sharey=True)
            # dic_axes = {i: axes[int(i / 4), int(i % 3)] for i in range(axes.size)}
            # dic_axes = {i: axes[int(i / 2), int(i % 2)] for i in range(axes.size)}

            for j, metric in enumerate(available_panels):
                # mask = np.array(data['scenario'] == scenario) & np.array(data['num_gt'] == num_gt) & np.array(data['method'] == method)

                mask1 = subset["error_rate"] == metric
                sub_subset = subset.loc[mask1]
                # for indel_r, error_r in list(itertools.product(*[INDEL_RATE_LST, error_rate_lst])):
                #     mask2 = np.logical_and(subset["indel_rate"] == indel_r, subset["error_rate"] == error_r)
                #     sub_subset.loc[mask2, "indel_rate__error_rate"] = f"{indel_r} - {error_r}"
                    # sub_subset = pd.concat([sub_subset, new_row], ignore_index=True)
                columns = [variables[0], "mafft_avg_sp_scores", variables[1]]
                title =  f"{dic_names['error_rate']} = {metric}"
                # title = " "
                xlabel = dic_names[variables[0]]
                ylabel = f'Average SP Score'
                # legend_title = 'IQtree or True Gene Trees'
                legend_title = dic_names[variables[1]]

                axes[0, j] = box_plot_generatoin(sub_subset, columns, axes[0, j], title, xlabel, ylabel, legend_title)
                axes[0, j].legend_.remove()
                axes[0, j].tick_params(axis='x', labelsize=11)
                axes[0, j].tick_params(axis='y', labelsize=11)
                axes[0, j].set_ylim(0.65, 1.1)
                axes[0, j].set_title(title, fontsize=9.5)
            # fig.suptitle(suptitle, fontsize=12)
            handles, labels = axes[0, j].get_legend_handles_labels()
            # fig.legend(handles, labels,title=legend_title, loc="upper center",  fontsize=11, title_fontsize=12, ncol=2)
            fig.legend(handles, labels, title=legend_title, loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=11, title_fontsize=12, ncol=2)
            plt.tight_layout()
            # plt.show()
            plt.savefig(save_addr, format="png", bbox_inches='tight')
            plt.close(fig)
            print(f"{idx}/{len(all_combinations)}")


def create_box_plot_iqtree_distance_for_paper(root_folder, results_folder, data, scenarioes, methods, num_species, num_gts, alignment_lengths, max_reticulations, iqtrees, maffts, num_real_reticulations, replicas, error_rate_lst, INDEL_RATE_LST):
    dic_names = {'scenario': "Scenario", "num_gt": "# Gene Trees", "method": "Method", "alignment_length": "Alignment Length",
     "error_rate": "Error Rate", "indel_rate":"Indel Rate", "iqtree":"IQTREE", "mafft":"MAFFT", "max_reticulation":"Max Reticulation"}
    # lst_parameters = {'scenario':scenarioes, "num_gt":num_gts, "method":methods, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
    #                   "indel_rate":INDEL_RATE_LST, "iqtree":iqtrees, "mafft":maffts, "max_reticulation":lst_max_reticulation}
    lst_parameters = {'scenario':scenarioes, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
                       "indel_rate":INDEL_RATE_LST}
    color_variables = [ "error_rate", "indel_rate"]
    x_axis_variables = ["alignment_length",]
    lst_masks = []
    count_all_figures = 0
    all_two_variables = list(itertools.product(*[x_axis_variables, color_variables]))
    for variables in all_two_variables:
        # if variables[0] != "method":
        #     continue
        print(variables)
        fixed_parameter_lst = [i for i in lst_parameters.keys() if i not in variables]
        # fixed_parameter_lst = [i for i in  if i not in variables]

        assert len(fixed_parameter_lst) == 2
        all_combinations = list(itertools.product(*[lst_parameters[i] for i in fixed_parameter_lst]))
        count_all_figures += len(all_combinations)
        for idx, combo in enumerate(all_combinations):
            # df[cond1 & cond2 & cond3 & cond4 & cond5]
            # temp = np.array(list(zip(fixed_parameter_lst+ ["iqtree"], list(combo) + [True])))
            temp = np.array(list(zip(fixed_parameter_lst+ ["iqtree", "method", "max_reticulation"], list(combo) + [True, 'InferNetwork_MPL', np.unique(data["max_reticulation"])[0]])))

            all_params = {j:"Varies" for j in variables }
            for key, value in temp:
                all_params[key] = value
            mask = [1] * len(data)
            for i, item in enumerate(temp):
                if temp[i, 0] == "num_gt" or temp[i, 0] =="alignment_length" or  temp[i, 0] =="max_reticulation":
                    val = int(temp[i, 1])
                elif temp[i, 0] =="error_rate" or temp[i, 0] =="indel_rate":
                    val = float(temp[i, 1])
                elif temp[i, 0] =="iqtree" or temp[i, 0] =="mafft":
                    if temp[i, 1]=="True":
                        val = True
                    else:
                        val=False
                else:
                    val =  temp[i, 1]
                mask = mask & np.array(data[temp[i, 0]] == val)
                # print()
                # data.iloc[mask.astype(bool)]
            # mask = np.array(data['scenario'] == scenario) & np.array(data['num_gt'] == num_gt) & np.array(data['method'] == method)
            lst_masks.append(mask)
            assert sum(lst_masks[0]) == len(replicas) * len(num_gts) * len(alignment_lengths) * len(error_rate_lst) * len(maffts)
            suptitle = f'iqtree_RFdistance__'
            for key in lst_parameters.keys():
                suptitle += f'{key}:{all_params[key]}__'
            suptitle = suptitle[:-2]
            save_addr = f"{results_folder}paper/iqtree_RFdistance/{all_params['scenario']}/{'--'.join(variables)}/"
            os.makedirs(save_addr, exist_ok=True)
            save_addr +=  f"{suptitle}.png"
            if os.path.isfile(save_addr):
                continue
            suptitle = f"Average Normalized RF Distance for Different {dic_names[variables[0]]} and Different {dic_names[variables[1]]}"
            subset = data.iloc[mask.astype(bool)]
            num_panels = 0

            num_panels = 2
            available_panels = ["norm_avg_dis_truth_vs_mafft_false", "norm_avg_dis_truth_vs_mafft_true"]
            dic_distances = {"norm_avg_dis_truth_vs_mafft_false": "Average RF Distance", "norm_avg_dis_truth_vs_mafft_true":"Average RF Distance"}
            if num_panels == 0:
                continue
            else:
                fig, axes = plt.subplots(1, num_panels, figsize=(num_panels * 4, 6), squeeze=False, dpi=300, sharey=True)
            # dic_axes = {i: axes[int(i / 4), int(i % 3)] for i in range(axes.size)}
            # dic_axes = {i: axes[int(i / 2), int(i % 2)] for i in range(axes.size)}
            top_lim = np.max(subset[available_panels]) +  0.05
            bottom_lim = np.min(subset[available_panels]) - 0.05
            for j, metric in enumerate(available_panels):
                # mask = np.array(data['scenario'] == scenario) & np.array(data['num_gt'] == num_gt) & np.array(data['method'] == method)

                columns = [variables[0], metric, variables[1]]
                # title = f'Number of Gene Trees: {num_gt}'
                subset1 = subset.copy()
                subset1[variables[1]] =  subset1[variables[1]].astype(str)
                custom_palette = {f"{float(lst_parameters[variables[1]][0])}": '#1f77b4', f"{float(lst_parameters[variables[1]][1])}":'#ff7f0e', f"{float(lst_parameters[variables[1]][2])}":'#2ca02c'}

                if metric=="norm_avg_dis_truth_vs_mafft_false":
                    title = "Ground Truth vs. Simulated MSA-Inferred Gene Trees"
                elif metric=="norm_avg_dis_truth_vs_mafft_true":
                    title = "Ground Truth vs. MAFFT MSA-Inferred Gene Trees"
                xlabel = dic_names[variables[0]]
                ylabel = f'{dic_distances[metric]}'
                # legend_title = 'IQtree or True Gene Trees'
                legend_title = dic_names[variables[1]]
                axes[0, j] = box_plot_generation_with_palette(subset1, columns, axes[0, j], title, xlabel, ylabel, legend_title, custom_palette)
                axes[0, j].set_ylim(top=top_lim)
                axes[0, j].set_ylim(bottom=bottom_lim)
                axes[0, j].tick_params(axis='x', labelsize=11)
                axes[0, j].tick_params(axis='y', labelsize=11)
                axes[0, j].legend_.remove()
                axes[0, j].set_ylim(0, 0.35)
                axes[0, j].set_title(title, fontsize=9.5)

            # fig.suptitle(suptitle, fontsize=12)
            handles, labels = axes[0, j].get_legend_handles_labels()
            fig.legend(handles, labels,title=legend_title, loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=11, title_fontsize=12, ncol=2)
            plt.tight_layout()
            # plt.show()
            plt.savefig(save_addr, format="png", bbox_inches='tight')
            plt.close(fig)
            print(f"{idx}/{len(all_combinations)}")


def create_line_plot_kl_divergence_accumulate_for_paper(root_folder, results_folder, data, scenarioes, methods, num_species, num_gts, alignment_lengths, max_reticulations, iqtrees, maffts, num_real_reticulations, replicas, error_rate_lst, INDEL_RATE_LST):
    dic_names = {'scenario': "Scenario", "num_gt": "# Gene Trees", "method": "Method", "alignment_length": "Alignment Length",
     "error_rate": "Error Rate", "indel_rate":"Indel Rate", "iqtree":"IQTREE", "mafft":"MAFFT", "max_reticulation":"Max Reticulation"}
    # lst_parameters = {'scenario':scenarioes, "num_gt":num_gts, "method":methods, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
    #                   "indel_rate":INDEL_RATE_LST, "iqtree":iqtrees, "mafft":maffts, "max_reticulation":lst_max_reticulation}
    lst_parameters = {'scenario':scenarioes, "num_gt": num_gts, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
                       "indel_rate":INDEL_RATE_LST}
    color_variables = [ "indel_rate"]
    x_axis_variables = ["alignment_length"]
    lst_masks = []
    count_all_figures = 0
    all_two_variables = list(itertools.product(*[x_axis_variables, color_variables]))
    for variables in all_two_variables:
        print(variables)
        fixed_parameter_lst = [i for i in lst_parameters.keys() if i not in variables and i not in [ 'num_gt', 'error_rate', 'scenario']]
        # fixed_parameter_lst = [i for i in  if i not in variables]

        assert len(fixed_parameter_lst) == 0
        all_combinations = list(itertools.product(*[lst_parameters[i] for i in fixed_parameter_lst]))
        count_all_figures += len(all_combinations)
        for idx, combo in enumerate(all_combinations):
            # df[cond1 & cond2 & cond3 & cond4 & cond5]
            # temp = np.array(list(zip(fixed_parameter_lst+ ["iqtree"], list(combo) + [True])))
            temp = np.array(list(zip(fixed_parameter_lst+ ["iqtree", "method", "max_reticulation"], list(combo) + [True, 'MCMC_GT_pseudo', np.unique(data["max_reticulation"])[0]])))

            all_params = {j:"Varies" for j in variables }
            all_params['scenario'] = "Varies"
            all_params['num_gt'] ="Varies"
            all_params['error_rate'] ="Varies"
            for key, value in temp:
                all_params[key] = value
            mask = [1] * len(data)
            for i, item in enumerate(temp):
                if temp[i, 0] == "num_gt" or temp[i, 0] =="alignment_length" or  temp[i, 0] =="max_reticulation":
                    val = int(temp[i, 1])
                elif temp[i, 0] =="error_rate" or temp[i, 0] =="indel_rate":
                    val = float(temp[i, 1])
                elif temp[i, 0] =="iqtree" or temp[i, 0] =="mafft":
                    if temp[i, 1]=="True":
                        val = True
                    else:
                        val=False
                else:
                    val =  temp[i, 1]
                mask = mask & np.array(data[temp[i, 0]] == val)
                # print()
                # data.iloc[mask.astype(bool)]
            # mask = mask & np.array(data['error_rate'] == 0.1)
            lst_masks.append(mask)
            # assert sum(lst_masks[0]) == len(replicas) * len(num_gts) * len(alignment_lengths) * len(error_rate_lst) * len(maffts)
            suptitle = f'iqtree_KLdiver__'
            for key in lst_parameters.keys():
                suptitle += f'{key}:{all_params[key]}__'
            suptitle = suptitle[:-2]
            save_addr = f"{results_folder}paper/iqtree_KLdiver/{all_params['scenario']}/{'--'.join(variables)}/"
            os.makedirs(save_addr, exist_ok=True)
            save_addr +=  f"{suptitle}.png"
            if os.path.isfile(save_addr):
                continue
            suptitle = f"Average Normalized RF Distance for Different {dic_names[variables[0]]} and Different {dic_names[variables[1]]}"
            subset = data.iloc[mask.astype(bool)]
            num_panels = 0
            available_panels = []

            # kl_true_vs_original, kl_true_vs_estimated, _, l1_true_vs_original, l1_true_vs_estimated
            num_panels = 3
            available_panels = [[[0, 's0_0_ret', "kl_true_vs_original"], [0, 's0_0_ret', "kl_true_vs_estimated"], [0.1, 's0_0_ret', "kl_true_vs_estimated"]],
                                [[0, 's1_1_ret_down', "kl_true_vs_original"], [0, 's1_1_ret_down', "kl_true_vs_estimated"], [0.1, 's1_1_ret_down', "kl_true_vs_estimated"]],
                                [[0, 's4_2_ret_cross', "kl_true_vs_original"], [0, 's4_2_ret_cross', "kl_true_vs_estimated"], [0.1, 's4_2_ret_cross', "kl_true_vs_estimated"]]]
            dic_distances = {"kl_true_vs_original": "KL Divergence", "kl_true_vs_estimated":"KL Divergence"}
            if num_panels == 0:
                continue
            else:
                fig, axes = plt.subplots(3, num_panels, figsize=(num_panels * 4, 12), squeeze=False, dpi=300, sharey=True, sharex=True)
                dic_axes = {i: axes[int(i / 3), int(i % 3)] for i in range(axes.size)}
            # dic_axes = {i: axes[int(i / 2), int(i % 2)] for i in range(axes.size)}
            top_lim = np.max(subset[["kl_true_vs_original", "kl_true_vs_estimated"]]) +  0.05
            bottom_lim = np.min(subset[["kl_true_vs_original", "kl_true_vs_estimated"]]) - 0.05
            for i, row in enumerate(available_panels):
                for j, (err, scen, metric) in enumerate(row):

                    mask1 = np.array(subset['error_rate'] == err) & np.array(subset['scenario'] == scen)
                    subset1 = subset.copy()
                    subset1 = subset1.iloc[mask1.astype(bool)]
                    subset1[metric] = subset1[metric] * subset1["num_gt"]
                    subset1 = subset1.groupby(["replica", "alignment_length", "indel_rate", "error_rate"]).sum()
                    subset1[metric] = subset1[metric] / (subset1["num_gt"]/2)
                    subset1 = subset1[metric].reset_index()
                    subset1[variables[1]] =  subset1[variables[1]].astype(str)
                    subset1[variables[0]] = subset1[variables[0]].astype(str)

                    columns = [variables[0], metric, variables[1]]
                    # title = f'Number of Gene Trees: {num_gt}'

                    custom_palette = {f"{float(lst_parameters[variables[1]][0])}": '#1f77b4', f"{float(lst_parameters[variables[1]][1])}":'#ff7f0e', f"{float(lst_parameters[variables[1]][2])}":'#2ca02c'}

                    if metric=="kl_true_vs_original":
                        title = "Scenario II"
                    elif metric=="kl_true_vs_estimated":
                        if err==0:
                            title = "Scenario III"
                        elif err==0.1:
                            title = "Scenario IV(Err. rate 0.1)"

                    xlabel = dic_names[variables[0]]
                    ylabel = f'KL Divergence'
                    # legend_title = 'IQtree or True Gene Trees'
                    legend_title = dic_names[variables[1]]
                    axes[i, j] = line_plot_generation_with_palette(subset1, columns, axes[i, j], title, xlabel, ylabel, legend_title, custom_palette)
                    # axes[i, j].set_ylim(top=top_lim)
                    axes[i, j].set_ylim(top=1.25)
                    axes[i, j].set_ylim(bottom=bottom_lim)
                    axes[i, j].tick_params(axis='x', labelsize=11)
                    axes[i, j].tick_params(axis='y', labelsize=11)
                    axes[i, j].legend_.remove()
                    axes[i, j].set_title(title, fontsize=9.5)

            # fig.suptitle(suptitle, fontsize=12)
            handles, labels = axes[0, j].get_legend_handles_labels()
            fig.legend(handles, labels,title=legend_title, loc="upper left", bbox_to_anchor=(0.08, 0.97), fontsize=11, title_fontsize=12, ncol=3)
            plt.tight_layout()
            # plt.show()
            plt.savefig(save_addr, format="png", bbox_inches='tight')
            plt.close(fig)
            print(f"{idx}/{len(all_combinations)}")


def create_line_plot_bootstrap_for_paper(root_folder, results_folder, data, scenarioes, methods, num_species, num_gts, alignment_lengths, max_reticulations, iqtrees, maffts, num_real_reticulations, replicas, error_rate_lst, INDEL_RATE_LST):
    dic_names = {'scenario': "Scenario", "num_gt": "# Gene Trees", "method": "Method", "alignment_length": "Alignment Length",
     "error_rate": "Error Rate", "indel_rate":"Indel Rate", "iqtree":"IQTREE", "mafft":"MAFFT", "max_reticulation":"Max Reticulation"}
    # lst_parameters = {'scenario':scenarioes, "num_gt":num_gts, "method":methods, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
    #                   "indel_rate":INDEL_RATE_LST, "iqtree":iqtrees, "mafft":maffts, "max_reticulation":lst_max_reticulation}
    lst_parameters = {'scenario':scenarioes, "num_gt": num_gts, "alignment_length":alignment_lengths, "error_rate":error_rate_lst,
                       "indel_rate":INDEL_RATE_LST}
    color_variables = [ "indel_rate"]
    x_axis_variables = ["alignment_length"]
    lst_masks = []
    count_all_figures = 0
    all_two_variables = list(itertools.product(*[x_axis_variables, color_variables]))
    for variables in all_two_variables:
        print(variables)
        fixed_parameter_lst = [i for i in lst_parameters.keys() if i not in variables and i not in [ 'num_gt', 'error_rate', "scenario"]]
        # fixed_parameter_lst = [i for i in  if i not in variables]

        assert len(fixed_parameter_lst) == 0
        all_combinations = list(itertools.product(*[lst_parameters[i] for i in fixed_parameter_lst]))
        count_all_figures += len(all_combinations)
        for idx, combo in enumerate(all_combinations):
            # df[cond1 & cond2 & cond3 & cond4 & cond5]
            # temp = np.array(list(zip(fixed_parameter_lst+ ["iqtree"], list(combo) + [True])))
            temp = np.array(list(zip(fixed_parameter_lst+ ["iqtree", "method", "max_reticulation"], list(combo) + [True, 'MCMC_GT_pseudo', np.unique(data["max_reticulation"])[0]])))

            all_params = {j:"Varies" for j in variables }
            all_params['scenario'] = "Varies"
            all_params['num_gt'] ="Varies"
            all_params['error_rate'] ="Varies"
            for key, value in temp:
                all_params[key] = value
            mask = [1] * len(data)
            for i, item in enumerate(temp):
                if temp[i, 0] == "num_gt" or temp[i, 0] =="alignment_length" or  temp[i, 0] =="max_reticulation":
                    val = int(temp[i, 1])
                elif temp[i, 0] =="error_rate" or temp[i, 0] =="indel_rate":
                    val = float(temp[i, 1])
                elif temp[i, 0] =="iqtree" or temp[i, 0] =="mafft":
                    if temp[i, 1]=="True":
                        val = True
                    else:
                        val=False
                else:
                    val =  temp[i, 1]
                mask = mask & np.array(data[temp[i, 0]] == val)
                # print()
                # data.iloc[mask.astype(bool)]
            # mask = mask & np.array(data['error_rate'] == 0.1)
            lst_masks.append(mask)
            # assert sum(lst_masks[0]) == len(replicas) * len(num_gts) * len(alignment_lengths) * len(error_rate_lst) * len(maffts)
            suptitle = f'iqtree_bootstrap__'
            for key in lst_parameters.keys():
                suptitle += f'{key}:{all_params[key]}__'
            suptitle = suptitle[:-2]
            save_addr = f"{results_folder}paper/iqtree_bootstrap/{all_params['scenario']}/{'--'.join(variables)}/"
            os.makedirs(save_addr, exist_ok=True)
            save_addr +=  f"{suptitle}.png"
            if os.path.isfile(save_addr):
                continue
            suptitle = f"Average Normalized RF Distance for Different {dic_names[variables[0]]} and Different {dic_names[variables[1]]}"
            subset = data.iloc[mask.astype(bool)]
            num_panels = 0
            available_panels = []

            num_panels = 3
            available_panels = [[[0, 's0_0_ret', "bootstrap_true_vs_original_refinement"], [0, 's0_0_ret', "bootstrap_true_vs_estimated_refinement"], [0.1, 's0_0_ret', "bootstrap_true_vs_estimated_refinement"]],
                                [[0, 's1_1_ret_down', "bootstrap_true_vs_original_refinement"], [0, 's1_1_ret_down', "bootstrap_true_vs_estimated_refinement"], [0.1, 's1_1_ret_down', "bootstrap_true_vs_estimated_refinement"]],
                                [[0, 's4_2_ret_cross', "bootstrap_true_vs_original_refinement"], [0, 's4_2_ret_cross', "bootstrap_true_vs_estimated_refinement"], [0.1, 's4_2_ret_cross', "bootstrap_true_vs_estimated_refinement"]]]
            dic_distances = {"bootstrap_true_vs_original_refinement": "Refinement Percentage", "bootstrap_true_vs_estimated_refinement":"Refinement Percentage"}
            if num_panels == 0:
                continue
            else:
                fig, axes = plt.subplots(3, num_panels, figsize=(num_panels * 4, 12), squeeze=False, dpi=300, sharey=True, sharex=True)
                dic_axes = {i: axes[int(i / 3), int(i % 3)] for i in range(axes.size)}
            # dic_axes = {i: axes[int(i / 2), int(i % 2)] for i in range(axes.size)}
            top_lim = np.max(subset[["bootstrap_true_vs_original_refinement", "bootstrap_true_vs_estimated_refinement"]])
            bottom_lim = np.min(subset[["bootstrap_true_vs_original_refinement", "bootstrap_true_vs_estimated_refinement"]]) - 0.05
            for i, row in enumerate(available_panels):
                for j, (err, scen, metric) in enumerate(row):
                    mask1 = np.array(subset['error_rate'] == err) & np.array(subset['scenario'] == scen)
                    subset1 = subset.copy()
                    subset1 = subset1.iloc[mask1.astype(bool)]
                    subset1[metric] = subset1[metric] * subset1["num_gt"]
                    subset1 = subset1.groupby(["replica", "alignment_length", "indel_rate", "error_rate"]).sum()
                    subset1[metric] = subset1[metric] / (subset1["num_gt"]/2)
                    subset1 = subset1[metric].reset_index()
                    subset1[variables[1]] =  subset1[variables[1]].astype(str)
                    subset1[variables[0]] = subset1[variables[0]].astype(str)
                    columns = [variables[0], metric, variables[1]]
                    # title = f'Number of Gene Trees: {num_gt}'

                    custom_palette = {f"{float(lst_parameters[variables[1]][0])}": '#1f77b4', f"{float(lst_parameters[variables[1]][1])}":'#ff7f0e', f"{float(lst_parameters[variables[1]][2])}":'#2ca02c'}

                    if metric=="bootstrap_true_vs_original_refinement":
                        title = "Scenario II"
                    elif metric=="bootstrap_true_vs_estimated_refinement":
                        if err==0:
                            title = "Scenario III"
                        elif err==0.1:
                            title = "Scenario IV(Err. rate 0.1)"

                    xlabel = dic_names[variables[0]]
                    ylabel = f'{dic_distances[metric]}'
                    # legend_title = 'IQtree or True Gene Trees'
                    legend_title = dic_names[variables[1]]
                    axes[i, j] = line_plot_generation_with_palette(subset1, columns, axes[i, j], title, xlabel, ylabel, legend_title, custom_palette)
                    axes[i, j].set_ylim(top=top_lim)
                    # axes[i, j].set_ylim(top=2)
                    axes[i, j].set_ylim(bottom=.75)
                    # axes[i, j].set_ylim(bottom=bottom_lim)
                    axes[i, j].tick_params(axis='x', labelsize=11)
                    axes[i, j].tick_params(axis='y', labelsize=11)
                    axes[i, j].legend_.remove()
                    axes[i, j].set_title(title, fontsize=9.5)

            # fig.suptitle(suptitle, fontsize=12)
            handles, labels = axes[0, j].get_legend_handles_labels()
            fig.legend(handles, labels,title=legend_title,  loc="lower left", bbox_to_anchor=(0.065, 0.06), fontsize=11, title_fontsize=12, ncol=3)
            plt.tight_layout()
            # plt.show()
            plt.savefig(save_addr, format="png", bbox_inches='tight')
            plt.close(fig)
            print(f"{idx}/{len(all_combinations)}")

def create_aln_SP_scores():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    _plot_df = data_df[~(data_df.mafft_avg_sp_scores.isna())] \
        .drop_duplicates(subset=["replica", "scenario", "alignment_length", "num_gt",
                                 "indel_rate", "error_rate", "mafft_avg_sp_scores"])[
        ["replica", "scenario", "alignment_length", "num_gt",
         "indel_rate", "error_rate", "mafft_avg_sp_scores"]].copy()
    _plot_df["SP_score"] = _plot_df["mafft_avg_sp_scores"] * _plot_df["num_gt"]
    _plot_df = _plot_df.groupby(["replica", "alignment_length", "indel_rate", "error_rate"]).sum()
    _plot_df["SP_score"] = _plot_df["SP_score"] / _plot_df["num_gt"]
    _plot_df = _plot_df["SP_score"].reset_index()
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    sns.lineplot(data=_plot_df[_plot_df.alignment_length == 200],
                 x="error_rate", y="SP_score", style="indel_rate", hue="indel_rate",
                 # errorbar=("pi", 100), err_style="bars",
                 markers=True, dashes=False, palette="deep",
                 ax=axs[0])
    sns.lineplot(data=_plot_df[_plot_df.alignment_length == 500],
                 x="error_rate", y="SP_score", style="indel_rate", hue="indel_rate",
                 # errorbar=("pi", 100), err_style="bars",
                 markers=True, dashes=False, palette="deep",
                 ax=axs[1], legend=False)
    sns.lineplot(data=_plot_df[_plot_df.alignment_length == 1000],
                 x="error_rate", y="SP_score", style="indel_rate", hue="indel_rate",
                 # errorbar=("pi", 100), err_style="bars",
                 markers=True, dashes=False, palette="deep",
                 ax=axs[2], legend=False)

    axs[0].set_title("Alignment length 200", fontsize=16)
    axs[1].set_title("Alignment length 500", fontsize=16)
    axs[2].set_title("Alignment length 1000", fontsize=16)

    axs[0].set_ylabel("SP score", fontsize=16)
    axs[0].set_xlabel("Error rate", fontsize=16)
    axs[1].set_xlabel("Error rate", fontsize=16)
    axs[2].set_xlabel("Error rate", fontsize=16)

    axs[0].set_yticks(np.arange(0.6, 1.01, 0.05))
    axs[0].set_yticklabels([f"{i:0.2f}" for i in np.arange(0.6, 1.01, 0.05)], fontsize=12)

    axs[0].set_xticks([0, 1, 2])
    axs[0].set_xticklabels(["0", "0.01", "0.10"], fontsize=12)
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticklabels(["0", "0.01", "0.10"], fontsize=12)
    axs[2].set_xticks([0, 1, 2])
    axs[2].set_xticklabels(["0", "0.01", "0.10"], fontsize=12)

    axs[0].legend(title="Indel rate", fontsize=12, title_fontsize=14, loc="lower left")

    plt.ylim((0.6, 1.01))

    plt.tight_layout()
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/aln_SP_scores-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()

def create_nRF_distances():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    _plot_df = data_df[~(data_df.norm_avg_dis_truth_vs_mafft_true.isna())].drop_duplicates(
        subset=["replica", "scenario", "alignment_length", "num_gt",
                "indel_rate", "error_rate", "norm_avg_dis_truth_vs_mafft_false", "norm_avg_dis_truth_vs_mafft_true"]
    )[["replica", "alignment_length", "num_gt",
       "indel_rate", "error_rate", "norm_avg_dis_truth_vs_mafft_false", "norm_avg_dis_truth_vs_mafft_true"]].copy()
    _plot_df["nRF_simul_aln"] = _plot_df["norm_avg_dis_truth_vs_mafft_false"] * _plot_df["num_gt"]
    _plot_df["nRF_mafft_aln"] = _plot_df["norm_avg_dis_truth_vs_mafft_true"] * _plot_df["num_gt"]
    _plot_df = _plot_df.groupby(["replica", "alignment_length", "indel_rate", "error_rate"]).sum()
    _plot_df["nRF_simul_aln"] = _plot_df["nRF_simul_aln"] / _plot_df["num_gt"]
    _plot_df["nRF_mafft_aln"] = _plot_df["nRF_mafft_aln"] / _plot_df["num_gt"]
    _plot_df = _plot_df[["nRF_simul_aln", "nRF_mafft_aln"]].reset_index()
    # _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True, sharex=True)

    sns.boxplot(data=_plot_df[_plot_df.error_rate == 0.],
                x="alignment_length", y="nRF_simul_aln", hue="indel_rate",
                palette="deep",
                ax=axs[0], boxprops=dict(edgecolor='none'))
    sns.boxplot(data=_plot_df[_plot_df.error_rate == 0.],
                x="alignment_length", y="nRF_mafft_aln", hue="indel_rate",
                palette="deep",
                ax=axs[1], legend=False, boxprops=dict(edgecolor='none'))
    sns.boxplot(data=_plot_df[_plot_df.error_rate == 0.1],
                x="alignment_length", y="nRF_mafft_aln", hue="indel_rate",
                palette="deep",
                ax=axs[2], legend=False,  boxprops=dict(edgecolor='none'))

    axs[0].set_title("Scenario II", fontsize=16)
    axs[1].set_title("Scenario III", fontsize=16)
    axs[2].set_title("Scenario IV (Err. rate 0.1)", fontsize=16)

    axs[0].set_ylabel("Avg. nRF distance", fontsize=16)
    axs[0].set_xlabel("Alignment length", fontsize=16)
    axs[1].set_xlabel("Alignment length", fontsize=16)
    axs[2].set_xlabel("Alignment length", fontsize=16)

    axs[0].set_yticks(np.arange(0., .351, 0.05))
    axs[0].set_yticklabels([f"{i:0.2f}" for i in np.arange(0., .351, 0.05)], fontsize=12)

    axs[0].set_xticks([0, 1, 2])
    axs[0].set_xticklabels(["200", "500", "1000"], fontsize=12)
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticklabels(["200", "500", "1000"], fontsize=12)

    axs[0].legend(title="Indel rate", fontsize=12, title_fontsize=14, loc="upper left")
    plt.ylim((0.02, .27))

    plt.tight_layout()
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/nRF_distances-dpi300.png", dpi=300, bbox_inches="tight")
    print()

def get_scenario(uses_mafft, uses_iqtree, has_error, error_rate):
    if not uses_mafft and not uses_iqtree and not has_error:
        return "I"
    elif not uses_mafft and uses_iqtree and not has_error:
        return "II"
    elif uses_mafft and uses_iqtree and not has_error:
        return "III"
    elif uses_mafft and uses_iqtree and has_error:
        if error_rate == "0.01":
            return "IV (0.01)"
        elif error_rate == "0.1":
            return "IV (0.10)"
        else:
            return np.nan
    else:
        return np.nan

def create_likelihoods_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s4_2_ret_cross"
    method = "InferNetwork_ML"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df["log_likelihood"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[").max())
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])
    _plot_df

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel("Log Likelihood", fontsize=16)
    axs[0].set_ylim((-1300, -650))
    axs[0].set_yticks(np.arange(-1200, -649, 100))
    axs[0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-1200, -649, 100)], fontsize=12)

    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([0, 1, 2, 3])
    axs[2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/likelihoods-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/likelihoods-dpi300.png", dpi=300, bbox_inches="tight")

    _plot_diff_df = _plot_df[
        ["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica", "max_reticulation",
         "log_likelihood"]] \
        .groupby(["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica",
                  "max_reticulation"]).sum().diff() \
        .where(lambda x: x > 0).dropna().reset_index().copy()
    _plot_diff_df = _plot_diff_df[_plot_diff_df.max_reticulation != "0"].copy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_diff_df[_plot_diff_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_diff_df[(_plot_diff_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel("Relative LLI increase", fontsize=16)
    axs[0].set_ylim((0, 275))
    axs[0].set_yticks(np.arange(0, 275, 50))
    axs[0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 275, 50)], fontsize=12)

    axs[0].set_xticks([-1, 0, 1, 2])
    axs[0].set_xticklabels(["", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([-1, 0, 1, 2])
    axs[1].set_xticklabels(["", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([-1, 0, 1, 2])
    axs[2].set_xticklabels(["", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="upper right")

    plt.tight_layout()
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/relative-lli-change-dpi300.png", dpi=300, bbox_inches="tight")
    print()

def create_2ret_likelihoods_lli_change_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s4_2_ret_cross"
    method = "InferNetwork_ML"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df["log_likelihood"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[").max())
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])
    _plot_df

    _plot_diff_df = _plot_df[
        ["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica", "max_reticulation",
         "log_likelihood"]] \
        .groupby(["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica",
                  "max_reticulation"]).sum().diff() \
        .where(lambda x: x > 0).dropna().reset_index().copy()
    _plot_diff_df = _plot_diff_df[_plot_diff_df.max_reticulation != "0"].copy()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[0][i], legend=plot_legend)

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_diff_df[_plot_diff_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_diff_df[(_plot_diff_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[1][i], legend=False)

    axs[0][0].set_ylabel("Log Likelihood", fontsize=16)
    axs[0][0].set_ylim((-1300, -650))
    axs[0][0].set_yticks(np.arange(-1200, -649, 100))
    axs[0][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-1200, -649, 100)], fontsize=12)

    axs[0][0].set_xticks([0, 1, 2, 3])
    axs[0][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][1].set_xticks([0, 1, 2, 3])
    axs[0][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][2].set_xticks([0, 1, 2, 3])
    axs[0][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0][0].set_xlabel("Max reticulations", fontsize=16)
    axs[0][1].set_xlabel("Max reticulations", fontsize=16)
    axs[0][2].set_xlabel("Max reticulations", fontsize=16)

    axs[0][0].set_title("Indel rate 0.00", fontsize=16)
    axs[0][1].set_title("Indel rate 0.05", fontsize=16)
    axs[0][2].set_title("Indel rate 0.10", fontsize=16)

    axs[1][0].set_ylabel("Relative LL increase", fontsize=16)
    axs[1][0].set_ylim((0, 275))
    axs[1][0].set_yticks(np.arange(0, 275, 50))
    axs[1][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 275, 50)], fontsize=12)

    axs[1][0].set_xticks([0, 1, 2, 3])
    axs[1][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][1].set_xticks([0, 1, 2, 3])
    axs[1][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][2].set_xticks([0, 1, 2, 3])
    axs[1][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[1][0].set_xlabel("Max reticulations", fontsize=16)
    axs[1][1].set_xlabel("Max reticulations", fontsize=16)
    axs[1][2].set_xlabel("Max reticulations", fontsize=16)

    axs[1][0].set_title("", fontsize=16)
    axs[1][1].set_title("", fontsize=16)
    axs[1][2].set_title("", fontsize=16)

    axs[0][0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/2ret-likelihoods-lli-change-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/2ret-likelihoods-lli-change-dpi300.png", dpi=300, bbox_inches="tight")
    print()

def create_1ret_likelihoods_lli_change_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s1_1_ret_down"
    method = "InferNetwork_ML"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df["log_likelihood"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[").max())
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])
    _plot_df

    _plot_diff_df = _plot_df[
        ["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica", "max_reticulation",
         "log_likelihood"]] \
        .groupby(["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica",
                  "max_reticulation"]).sum().diff() \
        .where(lambda x: x > 0).dropna().reset_index().copy()
    _plot_diff_df = _plot_diff_df[_plot_diff_df.max_reticulation != "0"].copy()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[0][i], legend=plot_legend)

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_diff_df[_plot_diff_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_diff_df[(_plot_diff_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[1][i], legend=False)

    axs[0][0].set_ylabel("Log Likelihood", fontsize=16)
    axs[0][0].set_ylim((-1150, -500))
    axs[0][0].set_yticks(np.arange(-1150, -500, 100))
    axs[0][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-1150, -500, 100)], fontsize=12)

    axs[0][0].set_xticks([0, 1, 2, 3])
    axs[0][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][1].set_xticks([0, 1, 2, 3])
    axs[0][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][2].set_xticks([0, 1, 2, 3])
    axs[0][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0][0].set_xlabel("Max reticulations", fontsize=16)
    axs[0][1].set_xlabel("Max reticulations", fontsize=16)
    axs[0][2].set_xlabel("Max reticulations", fontsize=16)

    axs[0][0].set_title("Indel rate 0.00", fontsize=16)
    axs[0][1].set_title("Indel rate 0.05", fontsize=16)
    axs[0][2].set_title("Indel rate 0.10", fontsize=16)

    axs[1][0].set_ylabel("Relative LL increase", fontsize=16)
    axs[1][0].set_ylim((0, 400))
    axs[1][0].set_yticks(np.arange(0, 400, 100))
    axs[1][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 400, 100)], fontsize=12)

    axs[1][0].set_xticks([0, 1, 2, 3])
    axs[1][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][1].set_xticks([0, 1, 2, 3])
    axs[1][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][2].set_xticks([0, 1, 2, 3])
    axs[1][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[1][0].set_xlabel("Max reticulations", fontsize=16)
    axs[1][1].set_xlabel("Max reticulations", fontsize=16)
    axs[1][2].set_xlabel("Max reticulations", fontsize=16)

    axs[1][0].set_title("", fontsize=16)
    axs[1][1].set_title("", fontsize=16)
    axs[1][2].set_title("", fontsize=16)

    axs[0][0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/1ret-likelihoods-lli-change-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/1ret-likelihoods-lli-change-dpi300.png", dpi=300, bbox_inches="tight")
    print()


def create_0ret_likelihoods_lli_change_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s0_0_ret"
    method = "InferNetwork_ML"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df["log_likelihood"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[").max())
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])
    _plot_df

    _plot_diff_df = _plot_df[
        ["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica", "max_reticulation",
         "log_likelihood"]] \
        .groupby(["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica",
                  "max_reticulation"]).sum().diff() \
        .where(lambda x: x > 0).dropna().reset_index().copy()
    _plot_diff_df = _plot_diff_df[_plot_diff_df.max_reticulation != "0"].copy()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[0][i], legend=plot_legend)

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_diff_df[_plot_diff_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_diff_df[(_plot_diff_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[1][i], legend=False)

    axs[0][0].set_ylabel("Log Likelihood", fontsize=16)
    axs[0][0].set_ylim((-850, -300))
    axs[0][0].set_yticks(np.arange(-850, -300, 100))
    axs[0][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-850, -300, 100)], fontsize=12)

    axs[0][0].set_xticks([0, 1, 2, 3])
    axs[0][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][1].set_xticks([0, 1, 2, 3])
    axs[0][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][2].set_xticks([0, 1, 2, 3])
    axs[0][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0][0].set_xlabel("Max reticulations", fontsize=16)
    axs[0][1].set_xlabel("Max reticulations", fontsize=16)
    axs[0][2].set_xlabel("Max reticulations", fontsize=16)

    axs[0][0].set_title("Indel rate 0.00", fontsize=16)
    axs[0][1].set_title("Indel rate 0.05", fontsize=16)
    axs[0][2].set_title("Indel rate 0.10", fontsize=16)

    axs[1][0].set_ylabel("Relative LL increase", fontsize=16)
    axs[1][0].set_ylim((0, 41))
    axs[1][0].set_yticks(np.arange(0, 41, 20))
    axs[1][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 41, 20)], fontsize=12)

    axs[1][0].set_xticks([0, 1, 2, 3])
    axs[1][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][1].set_xticks([0, 1, 2, 3])
    axs[1][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][2].set_xticks([0, 1, 2, 3])
    axs[1][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[1][0].set_xlabel("Max reticulations", fontsize=16)
    axs[1][1].set_xlabel("Max reticulations", fontsize=16)
    axs[1][2].set_xlabel("Max reticulations", fontsize=16)

    axs[1][0].set_title("", fontsize=16)
    axs[1][1].set_title("", fontsize=16)
    axs[1][2].set_title("", fontsize=16)

    axs[0][0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/0ret-likelihoods-lli-change-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/0ret-likelihoods-lli-change-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_2ret_pseudolikelihoods_plli_change_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s4_2_ret_cross"
    method = "InferNetwork_MPL"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df["log_likelihood"] = _plot_df["log_probability"]
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    _plot_diff_df = _plot_df[
        ["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica", "max_reticulation",
         "log_likelihood"]] \
        .groupby(["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica",
                  "max_reticulation"]).sum().diff() \
        .where(lambda x: x > 0).dropna().reset_index().copy()
    _plot_diff_df = _plot_diff_df[_plot_diff_df.max_reticulation != "0"].copy()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[0][i], legend=plot_legend)

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_diff_df[_plot_diff_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_diff_df[(_plot_diff_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[1][i], legend=False)

    axs[0][0].set_ylabel("Log Pseudolikelihood", fontsize=16)
    axs[0][0].set_ylim((-3100, -1850))
    axs[0][0].set_yticks(np.arange(-3100, -1850, 200))
    axs[0][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-3100, -1850, 200)], fontsize=12)

    axs[0][0].set_xticks([0, 1, 2, 3])
    axs[0][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][1].set_xticks([0, 1, 2, 3])
    axs[0][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][2].set_xticks([0, 1, 2, 3])
    axs[0][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0][0].set_xlabel("Max reticulations", fontsize=16)
    axs[0][1].set_xlabel("Max reticulations", fontsize=16)
    axs[0][2].set_xlabel("Max reticulations", fontsize=16)

    axs[0][0].set_title("Indel rate 0.00", fontsize=16)
    axs[0][1].set_title("Indel rate 0.05", fontsize=16)
    axs[0][2].set_title("Indel rate 0.10", fontsize=16)

    axs[1][0].set_ylabel("Relative PLL increase", fontsize=16)
    axs[1][0].set_ylim((0, 235))
    axs[1][0].set_yticks(np.arange(0, 235, 50))
    axs[1][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 235, 50)], fontsize=12)

    axs[1][0].set_xticks([0, 1, 2, 3])
    axs[1][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][1].set_xticks([0, 1, 2, 3])
    axs[1][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][2].set_xticks([0, 1, 2, 3])
    axs[1][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[1][0].set_xlabel("Max reticulations", fontsize=16)
    axs[1][1].set_xlabel("Max reticulations", fontsize=16)
    axs[1][2].set_xlabel("Max reticulations", fontsize=16)

    axs[1][0].set_title("", fontsize=16)
    axs[1][1].set_title("", fontsize=16)
    axs[1][2].set_title("", fontsize=16)

    axs[0][0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/2ret-pseudolikelihoods-plli-change-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/2ret-pseudolikelihoods-plli-change-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_1ret_pseudolikelihoods_plli_change_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s1_1_ret_down"
    method = "InferNetwork_MPL"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df["log_likelihood"] = _plot_df["log_probability"]
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    _plot_diff_df = _plot_df[
        ["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica", "max_reticulation",
         "log_likelihood"]] \
        .groupby(["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica",
                  "max_reticulation"]).sum().diff() \
        .where(lambda x: x > 0).dropna().reset_index().copy()
    _plot_diff_df = _plot_diff_df[_plot_diff_df.max_reticulation != "0"].copy()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[0][i], legend=plot_legend)

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_diff_df[_plot_diff_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_diff_df[(_plot_diff_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[1][i], legend=False)

    axs[0][0].set_ylabel("Log Pseudolikelihood", fontsize=16)
    axs[0][0].set_ylim((-2900, -1650))
    axs[0][0].set_yticks(np.arange(-2900, -1650, 200))
    axs[0][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-2900, -1650, 200)], fontsize=12)

    axs[0][0].set_xticks([0, 1, 2, 3])
    axs[0][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][1].set_xticks([0, 1, 2, 3])
    axs[0][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][2].set_xticks([0, 1, 2, 3])
    axs[0][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0][0].set_xlabel("Max reticulations", fontsize=16)
    axs[0][1].set_xlabel("Max reticulations", fontsize=16)
    axs[0][2].set_xlabel("Max reticulations", fontsize=16)

    axs[0][0].set_title("Indel rate 0.00", fontsize=16)
    axs[0][1].set_title("Indel rate 0.05", fontsize=16)
    axs[0][2].set_title("Indel rate 0.10", fontsize=16)

    axs[1][0].set_ylabel("Relative PLL increase", fontsize=16)
    axs[1][0].set_ylim((0, 325))
    axs[1][0].set_yticks(np.arange(0, 325, 50))
    axs[1][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 325, 50)], fontsize=12)

    axs[1][0].set_xticks([0, 1, 2, 3])
    axs[1][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][1].set_xticks([0, 1, 2, 3])
    axs[1][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][2].set_xticks([0, 1, 2, 3])
    axs[1][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[1][0].set_xlabel("Max reticulations", fontsize=16)
    axs[1][1].set_xlabel("Max reticulations", fontsize=16)
    axs[1][2].set_xlabel("Max reticulations", fontsize=16)

    axs[1][0].set_title("", fontsize=16)
    axs[1][1].set_title("", fontsize=16)
    axs[1][2].set_title("", fontsize=16)

    axs[0][0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/1ret-pseudolikelihoods-plli-change-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/1ret-pseudolikelihoods-plli-change-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_0ret_pseudolikelihoods_plli_change_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s0_0_ret"
    method = "InferNetwork_MPL"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df["log_likelihood"] = _plot_df["log_probability"]
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    _plot_diff_df = _plot_df[
        ["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica", "max_reticulation",
         "log_likelihood"]] \
        .groupby(["scenario", "method", "alignment_length", "num_gt", "indel_rate", "error_rate", "replica",
                  "max_reticulation"]).sum().diff() \
        .where(lambda x: x > 0).dropna().reset_index().copy()
    _plot_diff_df = _plot_diff_df[_plot_diff_df.max_reticulation != "0"].copy()

    fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[0][i], legend=plot_legend)

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_diff_df[_plot_diff_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_diff_df[(_plot_diff_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[1][i], legend=False)

    axs[0][0].set_ylabel("Log Pseudolikelihood", fontsize=16)
    axs[0][0].set_ylim((-2300, -950))
    axs[0][0].set_yticks(np.arange(-2200, -950, 200))
    axs[0][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-2200, -950, 200)], fontsize=12)

    axs[0][0].set_xticks([0, 1, 2, 3])
    axs[0][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][1].set_xticks([0, 1, 2, 3])
    axs[0][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[0][2].set_xticks([0, 1, 2, 3])
    axs[0][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0][0].set_xlabel("Max reticulations", fontsize=16)
    axs[0][1].set_xlabel("Max reticulations", fontsize=16)
    axs[0][2].set_xlabel("Max reticulations", fontsize=16)

    axs[0][0].set_title("Indel rate 0.00", fontsize=16)
    axs[0][1].set_title("Indel rate 0.05", fontsize=16)
    axs[0][2].set_title("Indel rate 0.10", fontsize=16)

    axs[1][0].set_ylabel("Relative PLL increase", fontsize=16)
    axs[1][0].set_ylim((0, 65))
    axs[1][0].set_yticks(np.arange(0, 65, 20))
    axs[1][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 65, 20)], fontsize=12)

    axs[1][0].set_xticks([0, 1, 2, 3])
    axs[1][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][1].set_xticks([0, 1, 2, 3])
    axs[1][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1][2].set_xticks([0, 1, 2, 3])
    axs[1][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[1][0].set_xlabel("Max reticulations", fontsize=16)
    axs[1][1].set_xlabel("Max reticulations", fontsize=16)
    axs[1][2].set_xlabel("Max reticulations", fontsize=16)

    axs[1][0].set_title("", fontsize=16)
    axs[1][1].set_title("", fontsize=16)
    axs[1][2].set_title("", fontsize=16)

    axs[0][0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/0ret-pseudolikelihoods-plli-change-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/0ret-pseudolikelihoods-plli-change-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_2ret_mcmc_pl_MAP_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s4_2_ret_cross"
    method = "MCMC_GT_pseudo"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    # _plot_df["log_likelihoods"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[")[100:].mean())
    _plot_df["log_likelihoods"] = _plot_df["log_probability"]
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihoods", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel("MAP Log Pseudolikelihood", fontsize=16)
    axs[0].set_ylim((-3300, -1850))
    axs[0].set_yticks(np.arange(-3300, -1850, 200))
    axs[0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-3300, -1850, 200)], fontsize=12)

    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([0, 1, 2, 3])
    axs[2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/2ret-mcmc-pl-MAP-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/2ret-mcmc-pl-MAP-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_2ret_mcmc_pl_sample_std_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s4_2_ret_cross"
    method = "MCMC_GT_pseudo"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    # _plot_df["log_likelihoods"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[")[100:].mean())
    _plot_df["log_likelihood_std"] = _plot_df["likelihoods"].apply(
        lambda x: np.fromstring(x[1:-1], sep="];[")[100:].std())
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood_std", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel(r"Sample Log Pseudolikelihood $\sigma$", fontsize=16)
    axs[0].set_ylim((1.5, 5.8))
    axs[0].set_yticks(np.arange(1.5, 5.8, 0.5))
    axs[0].set_yticklabels([f"{i:0.1f}" for i in np.arange(1.5, 5.8, 0.5)], fontsize=12)

    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([0, 1, 2, 3])
    axs[2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/2ret-mcmc-pl-sample-std-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/2ret-mcmc-pl-sample-std-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_1ret_mcmc_pl_MAP_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s1_1_ret_down"
    method = "MCMC_GT_pseudo"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    # _plot_df["log_likelihoods"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[")[100:].mean())
    _plot_df["log_likelihoods"] = _plot_df["log_probability"]
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihoods", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel("MAP Log Pseudolikelihood", fontsize=16)
    axs[0].set_ylim((-3100, -1650))
    axs[0].set_yticks(np.arange(-3100, -1650, 200))
    axs[0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-3100, -1650, 200)], fontsize=12)

    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([0, 1, 2, 3])
    axs[2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/1ret-mcmc-pl-MAP-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/1ret-mcmc-pl-MAP-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_1ret_mcmc_pl_sample_std_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s1_1_ret_down"
    method = "MCMC_GT_pseudo"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    # _plot_df["log_likelihoods"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[")[100:].mean())
    _plot_df["log_likelihood_std"] = _plot_df["likelihoods"].apply(
        lambda x: np.fromstring(x[1:-1], sep="];[")[100:].std())
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood_std", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel(r"Sample Log Pseudolikelihood $\sigma$", fontsize=16)
    axs[0].set_ylim((1.5, 8.8))
    axs[0].set_yticks(np.arange(1.5, 8.8, 1.5))
    axs[0].set_yticklabels([f"{i:0.1f}" for i in np.arange(1.5, 8.8, 1.5)], fontsize=12)

    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([0, 1, 2, 3])
    axs[2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="upper left")

    plt.tight_layout()
    # plt.savefig("../plots/1ret-mcmc-pl-sample-std-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/1ret-mcmc-pl-sample-std-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()


def create_0ret_mcmc_pl_MAP_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s0_0_ret"
    method = "MCMC_GT_pseudo"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    # _plot_df["log_likelihoods"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[")[100:].mean())
    _plot_df["log_likelihoods"] = _plot_df["log_probability"]
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihoods", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel("MAP Log Pseudolikelihood", fontsize=16)
    axs[0].set_ylim((-2500, -950))
    axs[0].set_yticks(np.arange(-2500, -950, 200))
    axs[0].set_yticklabels([f"{i:0.0f}" for i in np.arange(-2500, -950, 200)], fontsize=12)

    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([0, 1, 2, 3])
    axs[2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="lower right")

    plt.tight_layout()
    # plt.savefig("../plots/0ret-mcmc-pl-MAP-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/0ret-mcmc-pl-MAP-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()

def create_0ret_mcmc_pl_sample_std_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    base_net = "s0_0_ret"
    method = "MCMC_GT_pseudo"

    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    _plot_df = data_df[(data_df.scenario == base_net) & (data_df.method == method)].copy()
    # _plot_df["log_likelihoods"] = _plot_df["likelihoods"].apply(lambda x: np.fromstring(x[1:-1], sep="];[")[100:].mean())
    _plot_df["log_likelihood_std"] = _plot_df["likelihoods"].apply(
        lambda x: np.fromstring(x[1:-1], sep="];[")[100:].std())
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["scenario"])

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey='row', sharex=True)

    indel_rates = ["0.0", "0.05", "0.1"]

    for i, indel_rate in enumerate(indel_rates):
        _plot_df_i = _plot_df[_plot_df.indel_rate == indel_rate]
        plot_legend = True
        if i > 0:
            _plot_df_i = pd.concat([_plot_df[(_plot_df.scenario == "I")], _plot_df_i], axis=0)
            plot_legend = False
        sns.lineplot(data=_plot_df_i,
                     x="max_reticulation", y="log_likelihood_std", style="scenario", hue="scenario",
                     errorbar=("ci", 50), err_style="band",
                     markers=True, dashes=False, palette="deep",
                     ax=axs[i], legend=plot_legend)

    axs[0].set_ylabel(r"Sample Log Pseudolikelihood $\sigma$", fontsize=16)
    axs[0].set_ylim((1.5, 8.8))
    axs[0].set_yticks(np.arange(1.5, 8.8, 1.5))
    axs[0].set_yticklabels([f"{i:0.1f}" for i in np.arange(1.5, 8.8, 1.5)], fontsize=12)

    axs[0].set_xticks([0, 1, 2, 3])
    axs[0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[1].set_xticks([0, 1, 2, 3])
    axs[1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2].set_xticks([0, 1, 2, 3])
    axs[2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[0].set_xlabel("Max reticulations", fontsize=16)
    axs[1].set_xlabel("Max reticulations", fontsize=16)
    axs[2].set_xlabel("Max reticulations", fontsize=16)

    axs[0].set_title("Indel rate 0.00", fontsize=16)
    axs[1].set_title("Indel rate 0.05", fontsize=16)
    axs[2].set_title("Indel rate 0.10", fontsize=16)

    axs[0].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="upper left")

    plt.tight_layout()
    # plt.savefig("../plots/0ret-mcmc-pl-sample-std-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/0ret-mcmc-pl-sample-std-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()

def create_topo_error_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    _plot_df = data_df[["scenario", "method", "max_reticulation",
                        "num_gt", "alignment_length", "error_rate", "indel_rate",
                        "mafft", "iqtree", "replica", "distance_RF", "distance_luay"]].copy()
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["inf_scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["inf_scenario"])
    _plot_df["topo_error"] = _plot_df[["distance_RF", "distance_luay"]].apply(
        lambda x: x["distance_luay"] if not np.isnan(x["distance_luay"]) else x["distance_RF"], axis=1)
    _plot_df

    fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharey='row', sharex=True)

    scenarios = ['s0_0_ret', 's1_1_ret_down', 's4_2_ret_cross']
    methods = ['InferNetwork_ML', 'InferNetwork_MPL', 'MCMC_GT_pseudo']
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            plot_legend = False
            if i == 0 and j == 2:
                plot_legend = True
            sns.lineplot(data=_plot_df[(_plot_df.scenario == scenario) & (_plot_df.method == method)],
                         x="max_reticulation", y="topo_error",
                         hue="inf_scenario",
                         palette="deep", hue_order=["I", "II", "III", "IV (0.01)", "IV (0.10)"],
                         err_style="band", errorbar=("ci", 50),
                         ax=axs[i][j], legend=plot_legend)

    axs[0][0].set_title("MLE", fontsize=16)
    axs[0][1].set_title("MPLE", fontsize=16)
    axs[0][2].set_title("MCMC MAP estimate", fontsize=16)

    axs[0][0].set_ylim((-0.1, 6.5))
    axs[0][0].set_yticks(np.arange(0, 6.5, 2))
    axs[0][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 6.5, 2)], fontsize=12)

    axs[1][0].set_ylim((-0.1, 6.5))
    axs[1][0].set_yticks(np.arange(0, 6.5, 2))
    axs[1][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 6.5, 2)], fontsize=12)

    axs[2][0].set_ylim((-0.1, 8.5))
    axs[2][0].set_yticks(np.arange(0, 8.5, 2))
    axs[2][0].set_yticklabels([f"{i:0.0f}" for i in np.arange(0, 8.5, 2)], fontsize=12)

    axs[0][0].set_ylabel("Topological error", fontsize=16)
    axs[1][0].set_ylabel("Topological error", fontsize=16)
    axs[2][0].set_ylabel("Topological error", fontsize=16)

    axs[2][0].set_xticks([0, 1, 2, 3])
    axs[2][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2][1].set_xticks([0, 1, 2, 3])
    axs[2][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2][2].set_xticks([0, 1, 2, 3])
    axs[2][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[2][0].set_xlabel("Max reticulations", fontsize=16)
    axs[2][1].set_xlabel("Max reticulations", fontsize=16)
    axs[2][2].set_xlabel("Max reticulations", fontsize=16)

    axs[0][2].legend(title="Scenario", fontsize=12, title_fontsize=14, loc="upper right")

    plt.tight_layout()
    # plt.savefig("../plots/topo-error-dpi300.png", dpi=300, bbox_inches="tight")
    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/topo-error-dpi300.png", dpi=300,
                bbox_inches="tight")
    print()

def create_topo_accuracy_bar_dpi300():
    data_df = pd.read_csv("/shared/mt100/6_book_chapter_final/results/final_result.csv")
    _plot_df = data_df[["scenario", "method", "max_reticulation",
                        "num_gt", "alignment_length", "error_rate", "indel_rate",
                        "mafft", "iqtree", "replica", "distance_RF", "distance_luay"]].copy()
    _plot_df["has_error"] = _plot_df["error_rate"].apply(lambda x: True if x > 0 else False)
    _plot_df["max_reticulation"] = _plot_df["max_reticulation"].astype(str)
    _plot_df["error_rate"] = _plot_df["error_rate"].astype(str)
    _plot_df["indel_rate"] = _plot_df["indel_rate"].astype(str)
    _plot_df["inf_scenario"] = _plot_df[["mafft", "iqtree", "has_error", "error_rate"]].apply(
        lambda x: get_scenario(x["mafft"], x["iqtree"], x["has_error"], x["error_rate"]), axis=1
    )
    _plot_df = _plot_df.dropna(subset=["inf_scenario"])
    _plot_df["topo_error"] = _plot_df[["distance_RF", "distance_luay"]].apply(
        lambda x: x["distance_luay"] if not np.isnan(x["distance_luay"]) else x["distance_RF"], axis=1)

    _table_df = _plot_df[["scenario", "method", "max_reticulation",
                          "num_gt", "alignment_length", "indel_rate",
                          "inf_scenario", "topo_error"]].groupby(
        ["scenario", "method", "max_reticulation",
         "num_gt", "alignment_length", "indel_rate",
         "inf_scenario"]).apply(lambda x: np.count_nonzero(x["topo_error"]) / len(x),
                                include_groups=False).reset_index().copy()
    _table_df["topo_accuracy"] = 1. - _table_df[[0]]
    fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharey='row', sharex=True)

    scenarios = ['s0_0_ret', 's1_1_ret_down', 's4_2_ret_cross']
    methods = ['InferNetwork_ML', 'InferNetwork_MPL', 'MCMC_GT_pseudo']
    for i, scenario in enumerate(scenarios):
        for j, method in enumerate(methods):
            plot_legend = False
            if i == 2 and j == 0:
                plot_legend = True
            sns.barplot(data=_table_df[(_table_df.scenario == scenario) & (_table_df.method == method) & \
                                       (_table_df.max_reticulation.astype(int) >= i)], x="max_reticulation",
                        y="topo_accuracy",
                        hue="inf_scenario",
                        palette="deep", hue_order=["I", "II", "III", "IV (0.01)", "IV (0.10)"],
                        errorbar=("pi", 50),
                        ax=axs[i][j], legend=plot_legend)

    axs[0][0].set_title("MLE", fontsize=16)
    axs[0][1].set_title("MPLE", fontsize=16)
    axs[0][2].set_title("MCMC MAP estimate", fontsize=16)

    axs[0][0].set_ylim((0, 1.1))
    axs[0][0].set_yticks(np.arange(0, 1.1, 0.2))
    axs[0][0].set_yticklabels([f"{i:0.1f}" for i in np.arange(0, 1.1, 0.2)], fontsize=12)

    axs[1][0].set_ylim((0, 1.1))
    axs[1][0].set_yticks(np.arange(0, 1.1, 0.2))
    axs[1][0].set_yticklabels([f"{i:0.1f}" for i in np.arange(0, 1.1, 0.2)], fontsize=12)

    axs[2][0].set_ylim((0, 1.1))
    axs[2][0].set_yticks(np.arange(0, 1.1, 0.2))
    axs[2][0].set_yticklabels([f"{i:0.1f}" for i in np.arange(0, 1.1, 0.2)], fontsize=12)

    axs[0][0].set_ylabel("Topological accuracy", fontsize=16)
    axs[1][0].set_ylabel("Topological accuracy", fontsize=16)
    axs[2][0].set_ylabel("Topological accuracy", fontsize=16)

    axs[2][0].set_xticks([0, 1, 2, 3])
    axs[2][0].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2][1].set_xticks([0, 1, 2, 3])
    axs[2][1].set_xticklabels(["0", "1", "2", "3"], fontsize=12)
    axs[2][2].set_xticks([0, 1, 2, 3])
    axs[2][2].set_xticklabels(["0", "1", "2", "3"], fontsize=12)

    axs[2][0].set_xlabel("Max reticulations", fontsize=16)
    axs[2][1].set_xlabel("Max reticulations", fontsize=16)
    axs[2][2].set_xlabel("Max reticulations", fontsize=16)

    axs[2][0].legend(title="Scenario", fontsize=10, title_fontsize=14, loc="upper left")

    plt.tight_layout()
    # plt.savefig("../plots/topo-accuracy-bar-dpi300.png", dpi=300, bbox_inches="tight")

    plt.savefig("/shared/mt100/6_book_chapter_final/results/paper/Nick_plots/topo-accuracy-bar-dpi300.png", dpi=300,
                bbox_inches="tight")

def main_inference():
    config_mafft(scenarios, root_folder, mafft_pkg, iqtree_folder, mafft_result_addr, INDEL_RATE_LST, error_rate_lst,
                 gap_penalty, cpu_cores, numbsim, start_replica)
    config_iqtree(scenarios, root_folder, iqtree_pkg, iqtree_folder, error_rate_lst, iqtree_result_addr, INDEL_RATE_LST,
                  cpu_cores, numbsim, start_replica)
    config_iqtree_bootstrap(scenarios, root_folder, iqtree_pkg, iqtree_folder, error_rate_lst, iqtree_result_addr,
                            INDEL_RATE_LST, cpu_cores, numbsim, start_replica)
    calc_RF_distance_gene_trees(scenarios, root_folder, error_rate_lst, INDEL_RATE_LST, numbsim, start_replica)
    calc_kl_divergence_l1_distance_gene_trees(scenarios, root_folder, error_rate_lst, INDEL_RATE_LST, numbsim,
                                              start_replica)
    config_infer_net(root_folder, scenarios, phylonet, num_tip_lst, numbsim, start_replica, num_gt_lst,
                     lst_max_reticulation, error_rate_lst, INDEL_RATE_LST, methods, params_mcmc_gt, cpu_cores)
    create_output(scenarios, root_folder, num_tip_lst, numbsim, start_replica, num_gt_lst, sites_per_gt_lst,
                  lst_max_reticulation, error_rate_lst, INDEL_RATE_LST, methods)


def main_ploting():
    ######## FOR Plots #######
    results_folder = f"{root_folder}results/"
    data, scenarioes, methods, num_species, num_gts, replicas, alignment_lengths, max_reticulations, iqtrees, maffts, num_real_reticulations = read_result_file(
        results_folder)
    create_box_plot_mafft_scores_for_paper(root_folder, results_folder, data, scenarioes, methods, num_species, num_gts,
                                           alignment_lengths, max_reticulations, iqtrees, maffts,
                                           num_real_reticulations, replicas, error_rate_lst, INDEL_RATE_LST)
    create_box_plot_iqtree_distance_for_paper(root_folder, results_folder, data, scenarioes, methods, num_species,
                                              num_gts, alignment_lengths, max_reticulations, iqtrees, maffts,
                                              num_real_reticulations, replicas, error_rate_lst, INDEL_RATE_LST)
    create_line_plot_kl_divergence_accumulate_for_paper(root_folder, results_folder, data, scenarioes, methods,
                                                        num_species, num_gts, alignment_lengths, max_reticulations,
                                                        iqtrees, maffts, num_real_reticulations, replicas,
                                                        error_rate_lst, INDEL_RATE_LST)
    create_line_plot_bootstrap_for_paper(root_folder, results_folder, data, scenarioes, methods, num_species, num_gts,
                                         alignment_lengths, max_reticulations, iqtrees, maffts, num_real_reticulations,
                                         replicas, error_rate_lst, INDEL_RATE_LST)
    create_aln_SP_scores()
    create_nRF_distances()
    create_likelihoods_dpi300()
    create_2ret_likelihoods_lli_change_dpi300()
    create_1ret_likelihoods_lli_change_dpi300()
    create_0ret_likelihoods_lli_change_dpi300()
    create_2ret_pseudolikelihoods_plli_change_dpi300()
    create_1ret_pseudolikelihoods_plli_change_dpi300()
    create_0ret_pseudolikelihoods_plli_change_dpi300()
    create_2ret_mcmc_pl_MAP_dpi300()
    create_2ret_mcmc_pl_sample_std_dpi300()
    create_1ret_mcmc_pl_MAP_dpi300()
    create_1ret_mcmc_pl_sample_std_dpi300()
    create_0ret_mcmc_pl_MAP_dpi300()
    create_0ret_mcmc_pl_sample_std_dpi300()
    create_topo_error_dpi300()
    create_topo_accuracy_bar_dpi300()


if __name__ == '__main__':
    root_folder = "/shared/mt100/6_book_chapter_final/"
    scenarios = ["s0_0_ret", "s1_1_ret_down", "s4_2_ret_cross"]
    num_tip_lst = [6]
    numbsim = 10
    start_replica = 0
    error_rate_lst = [0, 0.01, 0.1]
    num_gt_lst = [100, 250, 500]
    sites_per_gt_lst = [200, 500, 1000]
    INDEL_RATE_LST = [0, 0.05, 0.1]
    lst_max_reticulation = [0, 1, 2, 3]
    methods = ["InferNetwork_MPL", "MCMC_GT_pseudo", "InferNetwork_ML"]
    cpu_cores = int(psutil.cpu_count(logical=True) * .92)
    indelible_control_folder = root_folder + "INDELibleV1.03/"
    iqtree_folder = root_folder + "iqtree/"
    phylonet = root_folder + "PhyloNet.jar"
    iqtree_pkg = root_folder + "iqtree-2.3.5-Linux-intel/bin/iqtree2"
    mafft_pkg = "/shared/mt100/ml_env/bin/mafft"
    mafft_result_addr = root_folder + "iqtree/result_mafft.txt"
    iqtree_result_addr = root_folder + "iqtree/result.txt"
    FastS_addr = root_folder + "FastSP/FastSP.jar"
    params_mcmc_gt = {"cl": 1000000, "bl": 100000, "sf": 1000}
    gap_penalty = 1.2

    main_inference()
    main_ploting()



