from datasets import load_dataset
from collections import defaultdict
from dateutil import parser as dateparser  # 更强大的时间解析器
import json
import copy

def build_message_trees(dataset):
    message_trees = defaultdict(dict)
    children_map = defaultdict(list)
    id_to_node = {}
    for row in dataset:
        created_ts = int(dateparser.parse(row["created_date"]).timestamp() * 1_000_000)

        node = {
            "message_id": row["message_id"],
            "parent_id": row["parent_id"],
            "message_tree_id": row["message_tree_id"],
            "text": row["text"],
            "role": row["role"],
            "lang": row["lang"],
            "created_timestamp": created_ts,
            "children": []
        }
        id_to_node[row["message_id"]] = node

        if row["parent_id"] is not None:
            children_map[row["parent_id"]].append(node)

    for node in id_to_node.values():
        node_id = node["message_id"]
        if node_id in children_map:
            node["children"].extend(children_map[node_id])

    for node in id_to_node.values():
        if node["parent_id"] is None:
            tree_id = node["message_tree_id"]
            message_trees[tree_id] = node
    return message_trees

def generate_dialogue_paths(tree_id, node, path=None, path_indices=None):
    if path is None:
        path = []
    if path_indices is None:
        path_indices = []

    current_path = path + [{
        "text": node["text"],
        "role": node["role"],
        "lang": node["lang"],
        "created_timestamp": node["created_timestamp"]
    }]

    if not node["children"]:
        dialogue_id = f"{tree_id}-" + "-".join(map(str, path_indices))
        return [{
            "dialogue_id": dialogue_id,
            "multi_turn": True,
            "dialogue": current_path
        }]

    all_paths = []
    for i, child in enumerate(node["children"]):
        child_paths = generate_dialogue_paths(tree_id, child, current_path, path_indices + [i])
        all_paths.extend(child_paths)
    return all_paths

def generate_dialog_list(dataset):
    all_dialogues = []
    for tree_id, root in dataset.items():
        dialogues = generate_dialogue_paths(tree_id, root)
        all_dialogues.extend(dialogues)
        
    return all_dialogues

def flatten_dialogues_to_access_sequence(dialogues):
    access_seq = []
    for dialogue in dialogues:
        for idx, turn in enumerate(dialogue["dialogue"]):
            access_seq.append({
                "timestamp": turn["created_timestamp"],
                "dialogue_id": dialogue["dialogue_id"],
                "turn_index": idx,
                "role": turn["role"],
                "lang": turn['lang'],
                "text": turn["text"],
            })
    access_seq.sort(key=lambda x: x["timestamp"])
    return access_seq

def generate_openai_requests(access_sequence, model_name, duplicate=False):
    role_dict = {
        "prompter": "user",
        "assistant": "assistant",
    }

    prev_message = None
    all_reqs = []
    dialogues = {}
    for access in access_sequence:
        d_id = access['dialogue_id']
        if d_id not in dialogues:
            dialogues[d_id] = []

        dialogues[d_id].append({
            "role": role_dict[access['role']],
            "content": access['text']
        })

        this_message = dialogues[d_id]

        if duplicate:
            all_reqs.append({
                "model": f"{model_name}",
                "messages": copy.deepcopy(this_message),
                "temperature": 0.7,
                "max_tokens": 150,
                "top_p": 1.0,
                "n": 1,
                "stream": False
            })
        else:
            if prev_message is None or prev_message != this_message:
                all_reqs.append({
                    "model": f"{model_name}",
                    "messages": copy.deepcopy(this_message),
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "top_p": 1.0,
                    "n": 1,
                    "stream": False
                })
        
        prev_message = this_message
    
    return all_reqs

if __name__ == '__main__':
    duplicate = False
    model_name = 'DeepSeek-R1-Distill-Qwen-14B'
    ds = load_dataset("OpenAssistant/oasst1")

    res_message_trees = defaultdict(dict)
    res_dialogues = defaultdict(dict)
    res_access_sequence = defaultdict(dict)
    res_reqs = defaultdict(dict)

    mode = ['train', 'validation']
    # mode = ['validation']
    for m in mode:
        this_dataset = ds[m]
        message_trees = build_message_trees(this_dataset)
        dialogues = generate_dialog_list(message_trees)
        access_sequence = flatten_dialogues_to_access_sequence(dialogues)
        reqs = generate_openai_requests(access_sequence, model_name)
        res_message_trees[m] = message_trees
        res_dialogues[m] = dialogues
        res_access_sequence[m] = access_sequence
        res_reqs[m] = reqs

    with open("oasst1_trees.json", "w", encoding="utf-8") as f:
        json.dump(res_message_trees, f, indent=2, ensure_ascii=False)
    with open("oasst1_dialogs.json", "w", encoding="utf-8") as f:
        json.dump(res_dialogues, f, indent=2, ensure_ascii=False)
    with open("oasst1_sequence.json", "w", encoding="utf-8") as f:
        json.dump(res_access_sequence, f, indent=2, ensure_ascii=False)
    if duplicate:
        req_file_name = f"oasst1_reqs_{model_name}_duplicate.json"
    else:
        req_file_name = f"oasst1_reqs_{model_name}_unique.json"
    with open(req_file_name, "w", encoding="utf-8") as f:
        json.dump(res_reqs, f, indent=2, ensure_ascii=False)
