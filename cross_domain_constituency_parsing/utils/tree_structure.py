from typing import Union, List, Iterator, Optional, Tuple, Dict
import numpy as np
import math


PUNCTUATION_SET = {",", ":", "``", "\'\'", ".", "?", "!"}


class Tree(object):
    """Tree data structure.
    Attributes:
        label:
        word:
        is_leaf:
        left and right: span idx in sentence, left included, right included. -> [left, right]
    """

    def __init__(
        self, label: str, children_or_word: Union[List['Tree'], str], word_index: Optional[int] = None
    ) -> None:
        super(Tree, self).__init__()

        self.label: str = label
        self.is_leaf: bool = isinstance(children_or_word, str)

        if self.is_leaf:
            self.word: Optional[str] = children_or_word
            self.children: Optional[List[Tree]] = None
            self.left: int = word_index
            self.right: int = word_index
        else:
            self.word: Optional[str] = None
            self.children: Optional[List[Tree]] = children_or_word
            self.left: int = self.children[0].left
            self.right: int = self.children[-1].right

    def linearize(self) -> str:
        if self.is_leaf:
            text = self.word
        else:
            text = ' '.join([child.linearize() for child in self.children])
        return '(%s %s)' % (self.label, text)

    def leaves(self) -> Iterator[str]:
        if self.is_leaf:
            yield self.word
        else:
            for child in self.children:
                yield from child.leaves()

    def pos_tags(self) -> Iterator[str]:
        if self.is_leaf:
            yield self.label
        else:
            for child in self.children:
                yield from child.pos_tags()

    def __str__(self) -> str:
        return self.linearize()


def delete_punctuation(tree: Tree, position_shift: int = 0) -> Tuple[Optional[Tree], int]:
    if tree.is_leaf:
        if tree.label in PUNCTUATION_SET:
            return None, position_shift+1
        else:
            return Tree(tree.label, tree.word, tree.left-position_shift), position_shift
    else:
        children = []
        for child in tree.children:
            subtree, position_shift = delete_punctuation(child, position_shift)
            if subtree is not None:
                children.append(subtree)

        if len(children) == 0:
            return None, position_shift
        else:
            return Tree(tree.label, children), position_shift


def parse_bracketed_parse_tree(text: str) -> Tree:
    queue = text.replace("(", " ( ").replace(")", " ) ").split()
    queue.reverse()
    stack: List[Union[Tree, str]] = list()
    word_idx = -1

    while len(queue) > 0:
        token = queue.pop()

        if token != ")":
            stack.append(token)
        else:
            children = []
            while len(stack) > 0:
                subtree = stack.pop()
                if subtree != "(":
                    children.append(subtree)
                else:
                    break

            root_label = children.pop()
            children.reverse()
            if isinstance(children[0], str):
                assert len(children) == 1, "bracked parse tree parsing error"
                children = children[0]
                word_idx += 1
                tree = Tree(root_label, children, word_idx)
            else:
                assert all(isinstance(child, Tree) for child in children), "bracked parse tree parsing error"
                tree = Tree(root_label, children)
            stack.append(tree)

    assert len(stack) == 1 and len(queue) == 0, "bracked parse tree parsing error"
    tree = stack.pop()
    assert tree.linearize() == text, "bracked parse tree parsing error"
    return tree


def construct_bracketed_parse_tree(
    tree_label: np.ndarray, tree_split: np.ndarray,
    tokens: List[str], postags: List[str], vocab: Dict[int, str], start: int, end: int
) -> Tree:
    split = tree_split[start, end]
    label = vocab[tree_label[start, end]]

    if split == -1:
        assert start == end, "split error"
        children = [Tree(postags[start], tokens[start], start)]
    else:
        assert start <= split < end, "split error"
        children = [construct_bracketed_parse_tree(tree_label, tree_split, tokens, postags, vocab, start, split),
                    construct_bracketed_parse_tree(tree_label, tree_split, tokens, postags, vocab, split+1, end)]

    return Tree(label, children)


# 二叉化处理多叉情况必须深度优先前序遍历，否则有问题。
def binarization(tree: Tree) -> Tree:
    if tree.is_leaf:
        label = tree.label
        word = tree.word
        return Tree(label, word, tree.left)
    else:
        label = tree.label
        children = tree.children

        while len(children) == 1 and (not children[0].is_leaf):
            tree = tree.children[0]
            label = label + "::" + tree.label
            children = tree.children

        if len(children) > 2:
            left_child = children[0]
            right_child = Tree("*", children[1:])
            children = [left_child, right_child]

        children = [binarization(child) for child in children]

        return Tree(label, children)


# 反二叉化处理多叉情况必须深度优先后序遍历，否则有问题。
def debinarization(tree: Tree) -> Tree:
    if tree.is_leaf:
        label = tree.label
        word = tree.word
        return Tree(label, word, tree.left)
    else:
        label = tree.label
        children = tree.children

        if "::" in label:
            label_list = label.split("::")
            for i in range(len(label_list)-1, 0, -1):
                label = label_list[i]
                tree = Tree(label, children)
                children = [tree]
            label = label_list[0]

        children = [debinarization(child) for child in children]
        new_children = []
        for child in children:
            if child.label == "*":
                new_children.extend(child.children)
            else:
                new_children.append(child)
        children = new_children

        return Tree(label, children)


def get_tree_triples(tree: Tree) -> List[Tuple[int, int, str]]:
    if tree.is_leaf:
        triples = []
    else:
        triples = [(tree.left, tree.right, tree.label)]
        for child in tree.children:
            child_triples = get_tree_triples(child)
            triples.extend(child_triples)
    return triples


def match_boundary(bounds: List[int], poi: int):
    assert bounds[0] <= poi <= bounds[-1], "poi error"
    for i in range(len(bounds)):
        if poi == bounds[i]:
            return i
        if poi < bounds[i+1]:
            return i+0.5


def check_entity_in_tree(tree: Tree, start: int, end: int) -> bool:
    tree_left_bounds = [child.left for child in tree.children] + [tree.right]
    tree_right_bounds = [tree.left] + [child.right for child in tree.children]
    i = match_boundary(tree_left_bounds, start)
    j = match_boundary(tree_right_bounds, end)
    if isinstance(i, int) and isinstance(j, int):
        return True
    elif j - i < 1.:
        child_id = math.floor(i)
        return check_entity_in_tree(tree.children[child_id], start, end)
    else:
        return False


# load tree by recursive function
# def load_trees(path: str) -> List[Tree]:
#     trees = []
#     with open(path, 'r', encoding='utf-8') as reader:
#         for line in reader:
#             line = line.strip()

#             tree = generate_tree_from_str(line)
#             trees.append(tree)
#     return trees


# def generate_tree_from_str(text: str) -> Tree:
#     assert text.count('(') == text.count(')')
#     tokens = text.replace("(", " ( ").replace(")", " ) ").split()
#     idx = 0
#     return build_tree(tokens, idx, 0)[0]


# def build_tree(tokens: List[str], idx: int, span_startpoint_idx: int):
#     """generate a tree from tokens list.
#     and the tree to be generated is bracketed.
#     Args:
#         tokens[idx] must be '('
#         span_startpoint_idx: the start point idx in the sentence of the span
#     Returns:
#         tree and idx to be processed.
#     """
#     idx += 1
#     label = tokens[idx]
#     idx += 1
#     assert idx < len(tokens)

#     if tokens[idx] == '(':
#         children = []
#         span_endpoint_idx = span_startpoint_idx
#         while idx < len(tokens) and tokens[idx] == '(':
#             child, idx, span_endpoint_idx = build_tree(tokens, idx, span_endpoint_idx)
#             children.append(child)
#         # generate internal node
#         tree = Tree(label, children, span_startpoint_idx, span_endpoint_idx-1)
#         assert not tree.is_leaf
#     elif tokens[idx] == ')':
#         print('No word!!!')
#         exit(-1)
#     else:
#         word = tokens[idx]
#         idx += 1
#         # generate leaf node
#         span_endpoint_idx = span_startpoint_idx + 1
#         tree = Tree(label, word, span_startpoint_idx, span_endpoint_idx-1)
#         assert tree.is_leaf

#     assert tokens[idx] == ')'
#     return tree, idx+1, span_endpoint_idx


# if __name__ == "__main__":
#     with open("./data/origin/WSJ/train.txt", 'r', encoding="utf-8") as reader:
#         for line in reader:
#             tree = parse_bracketed_parse_tree(line.strip())
#             assert tree.linearize() == line.strip()
#             tree_ = debinarization(binarization(tree))
#             if tree_.linearize() != line.strip():
#                 print(tree_)
