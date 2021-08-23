from _typeshed import Self
from sys import prefix
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader

OUTPUT_DIM = 10
COMP_EMB_DIM = 20
WORD_EMB_DIM = 20
TAG_EMB_DIM = 5
VOCAB_SIZE = 10
N_COMP_NETWORKS = 4
TREE_DEPTH = 2


class TreeDataset(Dataset):
    def __init__(self) -> None:
        self.DUMMY_INPUT = {
            "2": {
                "tokens": [
                    ["The"],
                    ["complicated"],
                    ["language"],
                    ["in"],
                    ["the"],
                    ["huge"],
                    ["new"],
                    ["law"],
                    ["has"],
                    ["muddied"],
                    ["the"],
                    ["fight"],
                    ["."],
                ],
                "tags": [
                    "DT",
                    "JJ",
                    "NN",
                    "IN",
                    "DT",
                    "JJ",
                    "JJ",
                    "NN",
                    "VBZ",
                    "VBN",
                    "DT",
                    "NN",
                    ".",
                ],
                "targets": [
                    "B-NP",
                    "I-NP",
                    "E-NP",
                    "O",
                    "B-NP",
                    "I-NP",
                    "I-NP",
                    "E-NP",
                    "O",
                    "O",
                    "B-NP",
                    "E-NP",
                    "O",
                ],
            },
            "3": {
                "tokens": [
                    ["The", "complicated", "language"],
                    ["in"],
                    ["the", "huge", "new", "law"],
                    ["has"],
                    ["muddied"],
                    ["the", "fight"],
                    ["."],
                ],
                "tags": ["NP", "O", "NP", "O", "O", "NP", "O"],
                "targets": ["O", "B-PP", "E-PP", "O", "B-VP", "E-VP", "O"],
            },
            "4": {
                "tokens": [
                    ["The", "complicated", "language"],
                    ["in", "the", "huge", "new", "law"],
                    ["has"],
                    ["muddied", "the", "fight"],
                    ["."],
                ],
                "tags": ["O", "PP", "O", "VP", "O"],
                "targets": ["B-NP", "E-NP", "B-VP", "E-VP", "O"],
            },
            "5": {
                "tokens": [
                    [
                        "The",
                        "complicated",
                        "language",
                        "in",
                        "the",
                        "huge",
                        "new",
                        "law",
                    ],
                    ["has", "muddied", "the", "fight"],
                    ["."],
                ],
                "tags": ["NP", "VP", "O"],
                "targets": ["B-S", "I-S", "E-S"],
            },
            "6": {
                "tokens": [
                    [
                        "The",
                        "complicated",
                        "language",
                        "in",
                        "the",
                        "huge",
                        "new",
                        "law",
                        "has",
                        "muddied",
                        "the",
                        "fight",
                        ".",
                    ]
                ],
                "tags": ["S"],
                "targets": [],
            },
        }

        self.consituent_tags = (
            ["O", "NP", "VP", "PP", "S",]
            + [f"{prefix}-NP" for prefix in ("B", "I", "E", "S")]
            + [f"{prefix}-VP" for prefix in ("B", "I", "E", "S")]
            + [f"{prefix}-PP" for prefix in ("B", "I", "E", "S")]
            + [f"{prefix}-S" for prefix in ("B", "I", "E", "S")]
        )
        # TODO complete with full list of POS tags
        self.pos_tags = ["DT", "JJ", "NN", "IN", "VBZ", "VBN", "."]
        # TODO replace with actual vocab lookup
        self.vocab = [
            "The",
            "complicated",
            "language",
            "in",
            "the",
            "huge",
            "new",
            "law",
            "has",
            "muddied",
            "the",
            "fight",
            ".",
        ]

    def idx_to_label(self, idx, tag_type):
        if tag_type == "pos":
            return self.pos_tags[idx]
        return self.consituent_tags[idx]

    def label_to_idx(self, label, tag_type):
        if tag_type == "pos":
            return self.pos_tags.index(label)
        return self.consituent_tags.index(label)
    
    def idx_to_token(self, idx):
        return self.vocab[idx]
    
    def token_to_idx(self, token):
        return self.vocab.index(token)

    def __getitem__(self, index) -> Dict:
        ## this should return multiple levels of data for each item
        ## return type will be a dict, where the keys are levels and values are dicts with tokens and tags
        return self.DUMMY_INPUT


class CompositionalNetwork(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        vocab_size: int,
        word_emd_dim: int = 200,
        tag_emb_dim: int = 20,
        comp_emb_dim: int = 200,
        n_comp_layers: int = 4,
    ):
        self.word_emb_layer = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=word_emd_dim
        )
        self.tag_emb_layer = torch.nn.Embedding(
            num_embeddings=output_dim, embedding_dim=tag_emb_dim
        )
        self.compositional_layers = {
            k: torch.nn.Linear(
                in_features=(word_emd_dim + tag_emb_dim) * k, out_features=comp_emb_dim
            )
            for k in range(1, n_comp_layers + 1)
        }

    def identify_chunks(self, tags, level):
        """
        identify chunks using BIOES tags. Return a list of <chunk index, length> tuples
        """
        # if level == 1, then each token is a standalone chunk
        if level == 1:
            return [(i, 1) for i in range(len(tags))]
        chunks = list()
        current_chunk = {"start_index": -1, "length": 0}
        for i, tag in enumerate(tags):
            if tag == "O" or tag.split("-")[0] == "S":
                if current_chunk["start_index"] != -1:
                    chunks.append(
                        (current_chunk["start_index"], current_chunk["length"])
                    )
                chunks.append((i, 1))
                current_chunk = {"start_index": -1, "length": 0}
            else:
                # check if current tag starts with 'E'
                if tag.split("-")[0] == "E":
                    if current_chunk["start_index"] == -1:
                        current_chunk["start_index"] = i
                    chunks.append(
                        (current_chunk["start_index"], current_chunk["length"] + 1)
                    )
                    current_chunk = {"start_index": -1, "length": 0}
                elif tag.split("-")[0] == "B":
                    current_chunk = {"start_index": i, "length": 1}
                else:
                    current_chunk["length"] += 1
        if current_chunk["start_index"] != -1:
            chunks.append((current_chunk["start_index"], current_chunk["length"]))
        return chunks

    def forward(self, x: Dict):
        """
        :param x: a dict of tokens and tags. Tags are used to identify chunks, which decide which compositional layer to use.
        
        """
        chunks = self.identify_chunks(x["tags"], level=x["level"])
        if x["use_embedding"]:
            token_embeddings = self.word_emb_layer(x["token_indices"])
        else:
            token_embeddings = torch.vstack(x["composed_vectors"])
        tag_embeddings = self.tag_emb_layer(x["tag_indices"])
        # iterate through chunks, and pass each through appropriate compostional layer
        composed_embeddings = []
        for chunk_start_index, chunk_length in chunks:
            stacked_embeddings = torch.hstack(
                [
                    token_embeddings[
                        chunk_start_index : (chunk_start_index + chunk_length)
                    ],
                    tag_embeddings[
                        chunk_start_index : (chunk_start_index + chunk_length)
                    ],
                ]
            )
            composed_embeddings.append(
                self.compositional_layers[chunk_length](stacked_embeddings)
            )
        return torch.cat(composed_embeddings, dim=0)


class Tagger(torch.nn.Module):
    def __init__(self, output_dim: int, comp_emb_dim: int, rnn_dim: int = 128):
        self.recurrent_layer = torch.nn.LSTM(
            input_size=comp_emb_dim, hidden_size=rnn_dim, batch_first=True
        )
        self.output_layer = torch.nn.Linear(
            in_features=rnn_dim, out_features=output_dim
        )

    def forward(self, x):
        """
        forward pass. Input 'x' has shape <batch_size, sequence_length, embeddings>
        """
        rnn_output, _ = self.recurrent_layer(x)
        return self.output_layer(rnn_output)


def train_loop(
    loss_fn,
    optimizer,
    tagger_model: Tagger,
    comp_model: CompositionalNetwork,
    dataloader: DataLoader,
):
    """training loop"""
    # iterate through the dataset
    # each batch is a nested dict
    for batch in dataloader:
        # for each batch we work through entire tree
        temp_tagger_predictions = dict()
        temp_compositional_output = dict()
        for level in range(1, TREE_DEPTH + 1):
            # for first level, we use POS tags
            # TODO replace tokens with their index
            if level == 2:
                input_dict = {
                    "token_indices": [comp_model.token_to_idx(token) for token in batch[level]["tokens"]],
                    "tag_indices": [
                        comp_model.label_to_idx(tag, tag_type="pos")
                        for tag in batch[level]["tags"]
                    ],
                    "tags": batch[level]["tags"],
                    "target_indices": [
                        comp_model.label_to_idx(tag, tag_type="constituents")
                        for tag in batch[level]["targets"]
                    ],
                    "level": level,
                    "use_embedding": True,
                }

            # for other levels, we use predicted tags of previous level from the tagger model
            else:
                input_dict = {
                    "tokens": temp_compositional_output[level - 1],
                    "tag_indices": temp_tagger_predictions[level - 1],
                    "tags": [
                        comp_model.idx_to_label(idx)
                        for idx in temp_tagger_predictions[level - 1]
                    ],
                    "target_indices": [
                        comp_model.label_to_idx(tag, tag_type="constituents")
                        for tag in batch[level]["targets"]
                    ],
                    "level": level,
                    "use_embedding": False,
                }

            composed_output = comp_model(input_dict)
            tagger_output = torch.nn.LogSoftmax(tagger_model(composed_output))
            # TODO do inverse lookup for predictions to get text label from their indices, before storing them in temp_tagger_predictions
            optimizer.zero_grad()
            loss = loss_fn(tagger_output, torch.tensor(batch[level]["target_indices"]))
            loss.backward()
            optimizer.step()

            # store predictions of current level, for use in next level
            temp_tagger_predictions[level] = tagger_output
            temp_compositional_output[level] = composed_output


def main():
    """main function"""
    tagger_model = Tagger(output_dim=OUTPUT_DIM, comp_emb_dim=COMP_EMB_DIM)
    compositional_model = CompositionalNetwork(
        output_dim=OUTPUT_DIM,
        vocab_size=VOCAB_SIZE,
        word_emd_dim=WORD_EMB_DIM,
        comp_emb_dim=COMP_EMB_DIM,
        n_comp_layers=N_COMP_NETWORKS,
    )
    optim = torch.optim.Adam(
        params=[tagger_model.parameters(), compositional_model.parameters()]
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(Dataset(), batch_size=1)
    train_loop(
        loss_fn=loss_fn,
        dataloader=dataloader,
        tagger_model=tagger_model,
        comp_model=compositional_model,
        optimizer=optim,
    )

