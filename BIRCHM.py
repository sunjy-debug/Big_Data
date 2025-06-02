import numpy as np
from typing import Tuple
from pathlib import Path

class CFEntry:
    # Clustering Feature Entries
    def __init__(self, data: np.ndarray):
        self.N = 1
        self.LS = data.copy() # linear sum
        self.SS = data.copy() ** 2 # squared vector

    def centroid(self) -> np.ndarray:
        # return the centroid of the current clustering
        return self.LS / self.N

    def radius(self) -> float:
        # return the radius of the current clustering
        return np.sqrt(max(np.sum(self.SS) / self.N - np.sum((self.LS / self.N) ** 2), 0.0))
    
    def new_data_addition(self, data: np.ndarray):
        # add a new data point in the CFEntry and update N, LS, SS
        self.N += 1
        self.LS += data
        self.SS += data ** 2

    def CFEntry_merge(self, data: 'CFEntry'):
        # merge another CFEntry with the current CFEntry
        self.N += data.N
        self.LS += data.LS
        self.SS += data.SS

    def copy(self) -> 'CFEntry':
        # return a copy for re-assignment in spliting
        new = CFEntry(np.zeros_like(self.LS))
        new.N = self.N
        new.LS = self.LS.copy()
        new.SS = self.SS.copy()
        return new
    
class CFNode:
    # Clustering Feature Tree Node
    def __init__(self, threshold: float, max_entries: int, is_leaf: bool):
        # threshold: to decide whether the leaf node is going to merge
        # max_entries: the capacity of the node (leaf node: L, non-leaf node: B)
        # is_leaf: to decide whether it is a leaf node
        self.threshold = threshold
        self.max_entries = max_entries
        self.is_leaf = is_leaf

        self.entries = [] # CFEntry
        self.children = [] # CFNode

        self.next_leaf = None # the pointer to the next leaf node

    def closet_entry_search(self, data: np.ndarray) -> int:
        # return the index of the CFEntry of the closest data point to the current data point
        min_dist = float('inf') # the initial distance
        min_idx = -1 # the initial index
        for idx, entry in enumerate(self.entries):
            ctr = entry.centroid()
            dist = np.linalg.norm(data - ctr)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        return min_idx
    
    def new_entry_addition(self, entry: CFEntry, child_node: 'CFNode' = None):
        # insert a new CFEntry to the current node
        # if the number of entries < max_entries, insert it and append child_node in nodes if non-leaf node
        # otherwise, split the node
        self.entries.append(entry)
        if not self.is_leaf:
            self.children.append(child_node) # record the child_node pointer if non-leaf node

    def node_split(self) -> Tuple['CFNode', 'CFNode']:
        # if the number of entries reaches max_entries, we need to split the node
        entries_copy = [entry.copy() for entry, _ in list(zip(self.entries, self.children))]
        children_copy = [children.copy() for _, children in list(zip(self.entries, self.children))]

        # clear out the node
        self.entries = []
        self.children = []

        # find the two most distant CFEntries
        n = len(entries_copy)
        max_dist = -1
        data_a, data_b = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                ctr_i = entries_copy[i].centroid()
                ctr_j = entries_copy[j].centroid()
                dist = np.linalg.norm(ctr_i - ctr_j)
                if dist > max_dist:
                    max_dist = dist
                    data_a, data_b = i, j
        
        # initialize the two new left and right nodes
        left = CFNode(self.threshold, self.max_entries, self.is_leaf)
        right = CFNode(self.threshold, self.max_entries, self.is_leaf)
        # put the two data points in the left and right nodes
        left.entries.append(entries_copy[data_a])
        right.entries.append(entries_copy[data_b])

        # assign the rest of the data points based on the distance with the centroid
        for idx in range(n):
            if idx in set([data_a, data_b]):
                continue # ignore the two data points
            ctr_idx = entries_copy[idx]
            dist_a = np.linalg.norm(ctr_idx.centroid() - entries_copy[data_a].centroid())
            dist_b = np.linalg.norm(ctr_idx.centroid() - entries_copy[data_b].centroid())
            if dist_a < dist_b:
                left.entries.append(ctr_idx)
            else:
                right.entries.append(ctr_idx)
        # if it is non-leaf node, assign the childeren of the node
        if not self.is_leaf:
            self.children = []

            for idx in range(n):
                ctr_idx = entries_copy[idx]
                dist_a = np.linalg.norm(ctr_idx.centroid() - entries_copy[data_a].centroid())
                dist_b = np.linalg.norm(ctr_idx.centroid() - entries_copy[data_b].centroid())
                if dist_a < dist_b:
                    left.children.append(children_copy[idx])
                else:
                    right.children.append(children_copy[idx])
        
        return left, right
            

class BIRCH:
    def __init__(self, threshold: float, B: int, L: int):
        # threshold: the max_radius threshold of the CFEntry; if after the new node addition the radius > threshold, we cannot merge the node
        # B: the max_entries of non-leaf node
        # L: the max_entries of leaf node
        self.threshold = threshold
        self.B = B
        self.L = L

        # initialize the root node as the leaf node
        self.root = CFNode(threshold, L, is_leaf = True)
        self.leaf = self.root # on the initialization, there is only root node as leaf node

    def data_insertion(self, data: np.ndarray):
        # insert a data point in the CF Tree
        leaf = self._leaf_node_search(self.root, data)
        self._data_to_leaf_node_insertion(leaf, data)

    def _leaf_node_search(self, node: CFNode, data: np.ndarray) -> CFNode:
        # find the most appropriate leaf node for the data point
        # if the node is leaf-node, return; otherwise, find the nearest node and go down through the children nodes
        if node.is_leaf:
            return node
        
        idx = node.closet_entry_search(data)
        return self._leaf_node_search(node.children[idx], data)
    
    def _data_to_leaf_node_insertion(self, leaf: CFNode, data: np.ndarray):
        # insert one data point into the leaf node
        # if the leaf node is empty, we create a new CFEntry
        if len(leaf.entries) == 0:
            leaf.entries.append(CFEntry(data))
        else:
            # find the nearest node
            min_idx = leaf.closet_entry_search(data)
            min_entry = leaf.entries[min_idx]

            # merge
            entry_copy = min_entry.copy()
            entry_copy.new_data_addition(data)
            # if the radius is within the threshold, we can merge the data
            if entry_copy.radius() <= leaf.threshold:
                min_entry.new_data_addition(data)
            else:
                leaf.entries.append(CFEntry(data))

            # if the node exceed the capacity, we need to split the leaf node and merge upward
            if len(leaf.entries) > leaf.max_entries:
                self._node_split_upward(leaf)
        
    def _node_split_upward(self, node: CFNode):
        # if the node exceed the capacity, we need to split the leaf node and merge upward
        left_child, right_child = node.node_split()
        parent, parent_idx = self._parent_search(self.root, node)

        # when the node is root node, we create a new root node
        if parent is None:
            new_root = CFNode(self.threshold, self.B, is_leaf = False)
            new_root.entries = [left_child.entries[0].copy(), right_child.entries[0].copy()]
            new_root.children = [left_child, right_child]
            self.root = new_root

            # if the left and right nodes are leaf nodes, we update the leaf list
            if left_child.is_leaf and right_child.is_leaf:
                left_child.next_leaf = right_child
                right_child.next_leaf = node.next_leaf
                node.next_leaf = None
            return
    
        # if the node has parent nodes, we replace the the children nodes of it with new children nodes
        del parent.entries[parent_idx]
        del parent.children[parent_idx]

        parent.entries.insert(parent_idx + 1, right_child.entries[0].copy())
        parent.children.insert(parent_idx + 1, right_child)

        if left_child.is_leaf and right_child.is_leaf:
            left_child.next_leaf = right_child
            right_child.next_leaf = node.next_leaf
            node.next_leaf = None
        
        if len(parent.entries) > parent.max_entries:
            self._node_split_upward(parent)

    def _parent_search(self, current: CFNode, target: CFNode, parent: CFNode = None) -> Tuple[CFNode, int]:
        if current.is_leaf:
            return (None, -1)
        
        for idx, child in enumerate(current.children):
            if child is target:
                return (current, idx)
            rest = self._parent_search(child, target, current)
            if (rest[0] is not None) or (rest[0] is None and rest[1] != -1):
                return rest
        return (None, -1)
    
    def leaf_CFEntry(self) -> list:
        # return a list with all CFEntries in all leaf nodes
        leaves = []
        node = self.leaf
        while node is not None:
            for entry in node.entries:
                leaves.append(entry)
            node = node.next_leaf
        return leaves
    
    def sample(self, data: np.ndarray):
        leaf_CFentries = self.leaf_CFEntry()
        cluster_idxs = {entry: idx for idx, entry in enumerate(leaf_CFentries)}

        N, D = data.shape
        labels = np.empty(N)
        for i in range(N):
            x = data[i]
            leaf = self._leaf_node_search(self.root, x)

            if len(leaf.entries) == 0:
                labels[i] = -1
                continue

            min_idx = leaf.closet_entry_search(x)
            min_entry = leaf.entries[min_idx]
            labels[i] = cluster_idxs[min_entry]

        # save the labels
        out_dir = Path("outputs").parent
        out_dir.mkdir(exist_ok = True, parents = True)
        labels_path = out_dir / "birch_labels.csv"
        np.savetxt(labels_path, np.column_stack([np.arange(len(labels)), labels]), fmt='%d', delimiter=',', header='index,label', comments='')
        print(f"BIRCH labels written to {labels_path}.\n")

        #print the cluster summary
        unique, counts = np.unique(labels, return_counts = True)
        for idx, size in zip(unique, counts):
            print(f"Cluster {idx}: {size} points")