"""
parse syntax trees in Penn Treebank structure into a dict object, and save as a jsonl file
"""
from typing import Dict, List
from nltk.tree import Tree
import json
import argparse
import logging
from tqdm import tqdm

def add_missing_tokens(parse_dict: Dict) -> Dict:
    """
    Add tokens that do not merge at a given tree level and are hence ignored in the source data
    """
    for level, values in parse_dict.items():
        # skip the first level
        if level == 2:
            continue
        prev_level_tokens_w_tags = list(
            zip(parse_dict[level - 1]["tokens"], parse_dict[level - 1]["tags"])
        )
        # add a dummy item to prev_level_tokens_w_tags, to ensure all actual items are proccessed by the
        # while loop
        prev_level_tokens_w_tags += [("PADDING", "O")]
        prev_level_tokens_w_tags.reverse()
        logging.debug(f"fixing level {level}")
        chunk_index = 0
        missing_in_previous_chunk = False
        first_token = True
        current_token, current_pos_tag = prev_level_tokens_w_tags.pop()
        while prev_level_tokens_w_tags:
            if len(prev_level_tokens_w_tags) == 1 and missing_in_previous_chunk:
                logging.debug(f"inserting {current_token} in chunk: {chunk_index}")
                values["tokens"].insert(chunk_index, current_token)
                values["tags"].insert(chunk_index, "O")
            logging.debug(f"searching chunk: {chunk_index} for {current_token}")
            if all(t in values["tokens"][chunk_index] for t in current_token):
                missing_in_previous_chunk = False
                first_token = False
                current_token, current_pos_tag = prev_level_tokens_w_tags.pop()
                continue
            else:
                logging.debug(f"missing: {current_token}")
                if missing_in_previous_chunk or first_token:
                    logging.debug(f"inserting {current_token} in chunk: {chunk_index}")
                    values["tokens"].insert(chunk_index, current_token)
                    values["tags"].insert(chunk_index, "O")
                    missing_in_previous_chunk = False
                    current_token, current_pos_tag = prev_level_tokens_w_tags.pop()
                    if first_token:
                        chunk_index += 1
                    continue
                missing_in_previous_chunk = True
                chunk_index += 1

    return parse_dict


def _get_unique_target(targets: List) -> str:
    """
    Multi-token chunks have redudant labels, de-duplicate them and return single label
    appropriate for the whole chunk
    """
    bioes_values = [x.split("-")[0] for x in targets]
    syntax_tag = targets[0].split("-")[-1]
    if "B" in bioes_values:
        return f"B-{syntax_tag}"
    elif "E" in bioes_values:
        return f"E-{syntax_tag}"
    elif "I" in bioes_values:
        return f"I-{syntax_tag}"
    else:
        return syntax_tag


def fill_targets(parse_dict: Dict, max_level: int) -> Dict:
    """
    For each level in parse tree, add target labels using 'tags' of immediate higher level
    """
    for level, values in parse_dict.items():
        logging.debug(f"adding targets to level {level}")
        current_targets = list()
        if level == max_level:
            break
        for chunk, tag in zip(
            parse_dict[level + 1]["tokens"], parse_dict[level + 1]["tags"]
        ):
            if len(chunk) >= 2:
                granular_tags = [f"I-{tag}"] * len(chunk)
                granular_tags[0] = granular_tags[0].replace("I-", "B-")
                granular_tags[-1] = granular_tags[-1].replace("I-", "E-")
            else:
                granular_tags = [f"S-{tag}"]
            granular_tags = [
                x if x.split("-")[-1] != "O" else x.split("-")[-1]
                for x in granular_tags
            ]
            current_targets += granular_tags

        current_targets.reverse()
        # de-duplicate tags of chunks
        for chunk in values["tokens"]:
            # pop len(chunk) labels from current_targets
            # de-duplicate them and then add to values["targets"]
            targets = [current_targets.pop() for _ in range(len(chunk))]
            values["targets"].append(_get_unique_target(targets))

    return parse_dict


def build_parse_dict(src_tree: Tree) -> Dict:
    """
    Build a multi-lvel dict of treebank data, adding target tags for each level from tags of higher level
    """
    max_height = src_tree.height()
    parse_dict = {
        level: {"tokens": list(), "tags": list(), "targets": list()}
        for level in range(2, max_height)
    }
    # top most level is redudant..so we can stop 1 level before
    for level in range(2, max_height):
        for subtree in src_tree.subtrees(lambda t: t.height() == level):
            parse_dict[level]["tokens"].append(subtree.leaves())
            parse_dict[level]["tags"].append(subtree.label())

    # each level might be missing some unary tokens, add them back to 'tokens'
    # for completeness
    parse_dict = add_missing_tokens(parse_dict)

    # fill the tragets list of each level based on tags of next level
    parse_dict = fill_targets(parse_dict, max_level=src_tree.height() - 1)

    return parse_dict


def main(treebank_file, output_file):
    with open(treebank_file, encoding="utf-8") as infile, \
        open(output_file, "w", encoding="utf-8") as outfile:
        for line in tqdm(infile.readlines()):
            parse_tree = Tree.fromstring(line)
            parsed_dict = build_parse_dict(parse_tree)
            outfile.write(f"{json.dumps(parsed_dict)}\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        usage="python treebank_parser.py treebank_data json_output"
    )
    arg_parser.add_argument("treebank", help="path to file with treebank data")
    arg_parser.add_argument("json_output", help="output file to save parsed data")
    args = arg_parser.parse_args()
    main(args.treebank, args.json_output)
