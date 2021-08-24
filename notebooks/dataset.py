import json, logging
from nltk.tree import Tree
from tqdm import tqdm
from typing import Dict, List
################################################################################
def build_parsed_dict(src_tree: Tree) -> Dict:
    """ Takes Tree object as input, transforms it into a Dict object"""

    max_height_of_src_tree = src_tree.height()

    # 1. create empty Dict object of this max height
    parsed_dict = {
        level: {"tokens": list(), "tags": list(), "targets": list()}
        for level in range(2, max_height_of_src_tree)
    }

    # 2. Omitting the top most node (TOP) as it is redundant, let's fill the tokens and tags at each level
    for level in range(2, max_height_of_src_tree):
        for subtree in src_tree.subtrees(lambda t: t.height() == level):
            parsed_dict[level]["tokens"].append(subtree.leaves())
            parsed_dict[level]["tags"].append(subtree.label())

    # 3. Each level must be missing some unary tokens, add them back to 'tokens' for completeness
    parsed_dict = add_missing_tokens(parse_dict= parsed_dict)

    # 4. Fil the targets list of each level based on tags of next level
    parsed_dict = fill_targets(parsed_dict, max_level=src_tree.height()-1)

    return parsed_dict

################################################################################
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
        curr_level_tokens_w_tags = list(
            zip(parse_dict[level]["tokens"], parse_dict[level]["tags"])
        )
        # add a dummy item to prev_level_tokens_w_tags, to ensure all actual items are proccessed by the
        # while loop
        prev_level_tokens_w_tags += [("PADDING", "O")]
        # print(f"All Tags and Tokens from previous level for examination: {prev_level_tokens_w_tags}")
        prev_level_tokens_w_tags.reverse()
        # print(f"All Tags and Tokens from current level for examination: {curr_level_tokens_w_tags}")
        logging.debug(f"fixing level {level}")
        chunk_index = 0
        missing_in_previous_chunk = False
        first_token = True
        current_token, current_pos_tag = prev_level_tokens_w_tags.pop()
        # print(f"Let's check the current token {current_token} and the tag {current_pos_tag}")
        
        
        while prev_level_tokens_w_tags:
            if len(prev_level_tokens_w_tags) == 1 and missing_in_previous_chunk:
                logging.debug(f"inserting {current_token} in chunk: {chunk_index}")
                values["tokens"].insert(chunk_index, current_token)
                values["tags"].insert(chunk_index, "O")
            logging.debug(f"searching chunk: {chunk_index} for {current_token}")
            
            try:
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

            except IndexError:
                None
                # print("Out of Index error")
                # print(prev_level_tokens_w_tags)
                # print(values["tokens"])
                break
                
    return parse_dict
####################################################################    
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
            for _ in range(len(chunk)):
                try:
                    targets = current_targets.pop()
                except IndexError:
                    None
                    break
                    # print(len(chunk))
                    # print(current_targets)
            # targets = [current_targets.pop() for _ in range(len(chunk))]
            values["targets"].append(_get_unique_target(targets))

    return parse_dict
####################################################################
def main(input_dataset, output_dataset):

    with open(input_dataset, encoding="utf-8") as infile, \
        open(output_dataset, encoding="utf-8") as outfile:

        for idx,line in enumerate(tqdm(infile.readlines())):
            if idx<1 and idx>1:
                break
            sentence_to_tree = Tree.fromstring(line) #parses simple string to tree object
            parsed_dict = build_parsed_dict(sentence_to_tree)
        
        print(line)
        print(parsed_dict)


if __name__ == "__main__":

    input_file = "data/raw/sample_dataset.txt"
    output_file = "data/processed/processed_dataset.json"
    main(input_file,output_file)