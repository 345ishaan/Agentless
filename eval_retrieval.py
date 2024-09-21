"""
Runs retrieval evaluation on the given dataset.
"""

import ast
import re
import collections
from typing import List, Tuple, Optional, Dict
import pandas as pd
import re
import random
import argparse
import json
from multiprocessing import Pool
import os
import sys
from collections import defaultdict
from datasets import load_dataset
from typing import List
from agentless.util.preprocess_data import get_repo_files

PROJECT_FILE_LOC = "/Users/ig/Documents/AgentlessModal/Agentless/git_data_swe_verify_50/"

def load_json(filepath):
    return json.load(open(filepath, "r"))

def load_jsonl(filepath):
    """
    Load a JSONL file from the given filepath.

    Arguments:
    filepath -- the path to the JSONL file to load

    Returns:
    A list of dictionaries representing the data in each line of the JSONL file.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.structures = []

    def visit_ClassDef(self, node):
        self.structures.append(('class', node.name, node.lineno, node.end_lineno))
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        for parent in self.structures:
            if parent[0] == 'class' and parent[2] <= node.lineno <= parent[3]:
                self.structures.append(('function', f"{parent[1]}.{node.name}",
                                        node.lineno, node.end_lineno))
                break
        else:
            self.structures.append(('function', node.name, node.lineno, node.end_lineno))
        self.generic_visit(node)

def parse_file_structure(file_content: str) -> List[Tuple[str, str, int, int]]:
    """
    Parse the file content and return a list of tuples containing information about
    classes, methods, and functions.

    Args:
    file_content (str): The content of the file to parse.

    Returns:
    List[Tuple[str, str, int, int]]: A list of tuples, each containing
                                     (type, name, start_line, end_line).
    """
    tree = ast.parse(file_content)
    visitor = CodeVisitor()
    visitor.visit(tree)
    return visitor.structures

def find_structure_for_line(structures: List[Tuple[str, str, int, int]], line_number: int) -> Optional[Tuple[str, str, int, int]]:
    """
    Find the innermost structure (class, method, or function) that contains the given line number.

    Args:
    structures (List[Tuple[str, str, int, int]]): List of structures in the file.
    line_number (int): The line number to search for.

    Returns:
    Optional[Tuple[str, str, int, int]]: The structure containing the line, or None if not found.
    """
    matching_structures = [s for s in structures if s[2] <= line_number <= s[3]]
    return max(matching_structures, key=lambda s: s[2]) if matching_structures else None


def find_structure_for_lines(file_content: str, line_numbers: List[int]) -> Dict[int, Optional[Tuple[str, str, int, int]]]:
    """
    Find the structures (class, method, or function) for the given line numbers in a file.

    Args:
    file_path (str): Path to the file.
    line_numbers (List[int]): List of line numbers to search for.

    Returns:
    Dict[int, Optional[Tuple[str, str, int, int]]]: A dictionary mapping line numbers to their containing structures.
    """

    
    structures = parse_file_structure(file_content)
    return {line: find_structure_for_line(structures, line) for line in line_numbers}

def get_affected_files(patch_string):
    pattern = r'diff --git a/(.*?) b/(.*?)$'
    matches = re.findall(pattern, patch_string, re.MULTILINE)
    affected_files = set()
    for match in matches:
        affected_files.add(match[0])  # 'a' path
        affected_files.add(match[1])  # 'b' path
    
    return list(affected_files)

def get_affected_tags(bug, patch_string):
    """
    Returns a list of file_names and the lines that are affected in each file.
    """
    affected_lines = {}
    current_file = None
    current_line_number = 0

    # Regular expressions for parsing the patch
    file_pattern = re.compile(r'^diff --git a/(.*) b/.*$')
    hunk_pattern = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@')

    for line in patch_string.split('\n'):
        # Check if this line indicates a new file
        file_match = file_pattern.match(line)
        if file_match:
            current_file = file_match.group(1)
            affected_lines[current_file] = []
            continue

        # Check if this line is a hunk header
        hunk_match = hunk_pattern.match(line)
        if hunk_match:
            current_line_number = int(hunk_match.group(1)) - 1
            continue

        # Check if this line is an addition or modification
        if line.startswith('+') and not line.startswith('+++'):
            affected_lines[current_file].append((current_line_number, line[1:]))
            current_line_number += 1
        elif not line.startswith('-') and not line.startswith('---'):
            current_line_number += 1
    project_file = os.path.join(PROJECT_FILE_LOC, bug["instance_id"] + ".json")
    d_file = load_json(project_file)
    structure = d_file["structure"]
    file_contents = get_repo_files(structure, list(affected_lines.keys()))
    affected_tags = collections.defaultdict(list)
    for file_name, lines in affected_lines.items():
        # parse the file content
        file_content = file_contents[file_name]
        all_lines = [int(x[0]) for x in lines]
        tags_dict = find_structure_for_lines(file_content, all_lines)
        for line, tag in tags_dict.items():
            if tag is not None:
                tag_type = tag[0]
                tag_name = tag[1]
                if tag_type == "class":
                    affected_tags[file_name].append(f"class: {tag_name}")
                if tag_type == "function":
                    affected_tags[file_name].append(f"function: {tag_name}")

    return affected_tags



def get_retrieval_eval_results(swe_bench_data, pred_jsonl_path, top_k_list: List[int] = [1,3,5,7,10]):
    """
    Runs retrieval evaluation on the given dataset.
    """
    
    # read the jsonl file and for each instance_id, search the dataset for the instance_id and get the context
    # then run the retrieval evaluation on the context and the query
    # return the average precision, recall, and f1 score
    pred_data = load_jsonl(pred_jsonl_path)
    avg_recall = 0
    count = 0
    overall_tag_precision = 0
    overall_tag_recall = 0
    avg_recalls = {k : 0 for k in top_k_list}
    for pred in pred_data:
        instance_id = pred["instance_id"]
        lite_dataset = swe_bench_data.filter(lambda x: x["instance_id"] in [instance_id])
        if len(lite_dataset) == 0:
            continue
        gt_patch = lite_dataset[0]["patch"]
        pred_files = pred["found_files"]
        gt_files = get_affected_files(gt_patch)
        for _k in top_k_list:
            pred_files_k = pred_files[:_k]
            
            # get the recall.
            intersection = set(pred_files_k) & set(gt_files)
            recall_at_k = len(intersection) / len(set(gt_files))
            avg_recalls[_k] += recall_at_k
        if args.tag_pr:
            # get the list of gt functions which are proposed to change.
            affected_tags = get_affected_tags(lite_dataset[0], gt_patch)

            predicted_tags = collections.defaultdict(list)
            for file_name, related_locs in zip(pred["found_files"], pred["found_related_locs"]):
                for loc in related_locs:
                    predicted_tags[file_name].extend(loc.strip().split("\n"))
            # compute overall tag precision and recall
            matched_tags_cnt = 0
            gt_tags_cnt = 0
            pred_tags_cnt = 0
            for file_name, tags in predicted_tags.items():
                for tag in tags:
                    if tag in affected_tags[file_name]:
                        matched_tags_cnt += 1
                pred_tags_cnt += len(tags)
                gt_tags_cnt += len(affected_tags[file_name])
            tag_precision = matched_tags_cnt / pred_tags_cnt if pred_tags_cnt > 0 else 1.0
            tag_recall = matched_tags_cnt / gt_tags_cnt if gt_tags_cnt > 0 else 1.0
            overall_tag_precision += tag_precision
            overall_tag_recall += tag_recall
        count += 1
        

    if count == 0:
        print("No instances found in the dataset.")
        return {k: 0 for k in top_k_list}, 0
    # Calculate average recall for each k
    avg_recalls = {k: recall / count for k, recall in avg_recalls.items()}
    avg_tag_precision = overall_tag_precision / count
    avg_tag_recall = overall_tag_recall / count
    
    return avg_recalls, avg_tag_precision, avg_tag_recall, count


if __name__ == "__main__":
    # use parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="exploiter345/SWE-bench_Verified_50")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--preds_path", type=str, default="")
    parser.add_argument("--tag_pr", action="store_true")
    parser.add_argument("--no_tag_pr", action="store_false", dest="tag_pr")
    args = parser.parse_args()

    assert args.preds_path != ""

    # load the dataset 
    swe_bench_data = load_dataset(args.dataset_id, split=args.split_name)

    avg_recalls, avg_tag_precision, avg_tag_recall, count = get_retrieval_eval_results(swe_bench_data, args.preds_path)
    for k, recall in avg_recalls.items():
        print(f"Average recall@{k}: {recall:.4f}")
    print(f"Count: {count}")
    print(f"Average tag precision: {avg_tag_precision:.4f}")
    print(f"Average tag recall: {avg_tag_recall:.4f}")