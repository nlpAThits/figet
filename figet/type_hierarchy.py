
import json


PATH = "data/wordnet/type_hierarchy_7k.json"
ROOTS = {"Person100007846", "Group100031264", "Organization108008335", "Location100027167", "Time100028270",
         "Object100002684", "Event100029378", "Place108513718"}


class TypeHierarchy(object):
    def __init__(self, type_dict):
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

    def get_type_hierarchy(self):
        with open(PATH, "r") as f:
            data = f.read()
            return json.loads(data)

    def get_parents_id(self, type_id):
        """
        :param type_id:
        :return: list of the type_ids of the types up in the hierarchy
        """
        return self.hierarchy[type_id]
