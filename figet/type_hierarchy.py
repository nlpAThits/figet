
import json
from abc import ABC
from collections import defaultdict
from figet.evaluate import COARSE


class Hierarchy(ABC):
    def __init__(self):
        self.hierarchy = {}

    def get_parents_id(self, type_id):
        """
        :param type_id:
        :return: list of the type_ids of the types up in the hierarchy
        """
        return self.hierarchy[type_id]

    def remove_particular_case_of(self, hypernym, hyponym, root_ids, type_dict):
        """
        if the hypernym and the hyponym are "parents", it only keeps the hyponym
        """
        if hypernym in type_dict.label2idx and hyponym in type_dict.label2idx and \
                type_dict.label2idx[hypernym] in root_ids and type_dict.label2idx[hyponym] in root_ids:
            root_ids.remove(type_dict.label2idx[hypernym])

    def add_particular_case_of(self, type_to_add, type_a, root_ids, type_dict):
        """
        If type_a is present, then it also adds type_b as a root.
        """
        if type_a in type_dict.label2idx and type_to_add in type_dict.label2idx and type_dict.label2idx[type_a] in root_ids:
            root_ids.add(type_dict.label2idx[type_to_add])


# COARSE = {'person', 'group', 'organization', 'location', 'time', 'object', 'event', 'place'}

class BenultraHierarchy(Hierarchy):
    def __init__(self, type_dict):
        super().__init__()
        coarse = set(COARSE)
        coarse.remove("entity")
        wordnet_hierarchy = self.get_type_hierarchy()
        self.hierarchy = {}
        for label, my_id in type_dict.label2idx.items():
            if my_id in self.hierarchy:
                continue
            if label in coarse:
                self.hierarchy[my_id] = []
                continue

            parents = wordnet_hierarchy[label]
            root_ids = set()
            for parent in parents:
                if parent in coarse and parent in type_dict.label2idx:
                    root_ids.add(type_dict.label2idx[parent])

            self.remove_particular_case_of("group", "organization", root_ids, type_dict)
            self.remove_particular_case_of("object", "person", root_ids, type_dict)
            self.remove_particular_case_of("object", "location", root_ids, type_dict)
            self.remove_particular_case_of("object", "place", root_ids, type_dict)
            self.remove_particular_case_of("event", "time", root_ids, type_dict)

            if label == "country":
                root_ids.remove(type_dict.label2idx["organization"])
                # country is not a location, and it should
                root_ids.add(type_dict.label2idx["location"])

            # Place is also location but location is not place!!
            self.add_particular_case_of("place", "location", root_ids, type_dict)

            self.hierarchy[my_id] = list(root_ids)

    def get_type_hierarchy(self):
        path = "data/wordnet/benultra_hierarchy.txt"
        wordnet_hierarchy = defaultdict(set)
        with open(path, "r") as f:
            for line in f:
                item, parent = line.strip().split()
                wordnet_hierarchy[item].add(parent)
        return wordnet_hierarchy


class TypeHierarchy(Hierarchy):
    def __init__(self, type_dict):
        super().__init__()
        wordnet_hierarchy = self.get_type_hierarchy()
        self.hierarchy = {}
        for label, my_id in type_dict.label2idx.items():
            if my_id in self.hierarchy: continue
            parents = wordnet_hierarchy[label]
            root_ids = set()
            for path in parents:
                path = path.split("/")[1:]
                for item in path:
                    if item in ROOTS and item in type_dict.label2idx:
                        root_ids.add(type_dict.label2idx[item])

            self.remove_particular_case_of("Group100031264", "Organization108008335", root_ids, type_dict)
            self.remove_particular_case_of("Object100002684", "Person100007846", root_ids, type_dict)
            self.remove_particular_case_of("Object100002684", "Location100027167", root_ids, type_dict)
            self.remove_particular_case_of("Object100002684", "Place108513718", root_ids, type_dict)
            # Place is also location but location is not place!!
            self.add_particular_case_of("Place108513718", "Location100027167", root_ids, type_dict)

            self.hierarchy[my_id] = list(root_ids)

    def get_type_hierarchy(self):
        path = "data/wordnet/type_hierarchy_7k.json"
        with open(path, "r") as f:
            data = f.read()
            return json.loads(data)


class OntonotesTypeHierarchy(Hierarchy):
    def __init__(self, type_dict):
        super().__init__()
        self.hierarchy = {}
        for label in type_dict.label2idx:
            parents = label.split("/")[1:-1]
            parents_ids = []
            for i in range(1, len(parents) + 1):
                current = "/" + "/".join(parents[:i])
                parents_ids.append(type_dict.label2idx[current])
            my_id = type_dict.label2idx[label]
            self.hierarchy[my_id] = parents_ids
