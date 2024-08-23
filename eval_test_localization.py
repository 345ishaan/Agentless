import argparse
import os
import re
from ranked_repo_map import RepoMap
from datasets import load_dataset
from agentless.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    setup_logger,
)
from agentless.util.preprocess_data import (
    correct_file_paths,
    get_full_file_paths_and_classes_and_functions,
    get_repo_files,
    line_wrap_content,
    show_project_structure,
)

PROJECT_FILE_LOC = "/Users/ig/Documents/AgentlessModal/Agentless/git_data_swe_verify_50/"


def get_affected_files(patch_string):
    pattern = r'diff --git a/(.*?) b/(.*?)$'
    matches = re.findall(pattern, patch_string, re.MULTILINE)
    affected_files = set()
    for match in matches:
        affected_files.add(match[0])  # 'a' path
        affected_files.add(match[1])  # 'b' path
    
    return list(affected_files)

def get_test_loc_recall(data):
    count = 0
    recall = 0
    for bug in data:
        project_file = os.path.join(PROJECT_FILE_LOC, bug["instance_id"] + ".json")
        d = load_json(project_file)
        structure = d["structure"]
        patch_files = get_affected_files(bug["patch"])
        test_patch_files = get_affected_files(bug["test_patch"])
        if not len(test_patch_files):
            continue
        # now look at the structure
        rm = RepoMap(root="./", project_structure_cache=structure)
        all_files, _, __ = get_full_file_paths_and_classes_and_functions(structure)
        all_files = [f[0] for f in all_files]
        found_files = []
        print(bug["instance_id"], patch_files, test_patch_files)
        for file in patch_files:
            ranked_references = rm.find_references(file, all_files)
            found_files.extend(list(ranked_references.keys()))
        intersection = set(found_files) & set(test_patch_files)
        recall += len(intersection) / len(test_patch_files)
        count += 1
    return recall / count



if __name__ == "__main__":
    # use parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="exploiter345/SWE-bench_Verified_50")
    parser.add_argument("--split_name", type=str, default="test")
    parser.add_argument("--run_top_n", type=int, default=1)
    parser.add_argument("--repo_filter", type=str, default="")
    args = parser.parse_args()

    swe_bench_data = load_dataset(args.dataset_id, split=args.split_name)
    if args.repo_filter != "":
        swe_bench_data = swe_bench_data.filter(lambda x : x["repo"] == args.repo_filter)

    if args.run_top_n > 0:
        swe_bench_data = swe_bench_data.select(range(args.run_top_n))

    mean_recall_dict = get_test_loc_recall(swe_bench_data)
    print(mean_recall_dict)