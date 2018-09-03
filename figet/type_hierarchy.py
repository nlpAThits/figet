
class TypeHierarchy(object):
    def __init__(self, type_dict):
        self.hierarchy = {}
        for label in type_dict.label2idx:
            parents = label.split("/")[1:-1]
            parents_ids = []
            for i in range(1,len(parents) + 1):
                current = "/" + "/".join(parents[:i])
                parents_ids.append(type_dict.label2idx[current])
            my_id = type_dict.label2idx[label]
            self.hierarchy[my_id] = parents_ids

    def get_parents_id(self, type_id):
        """
        :param type_id:
        :return: list of the type_ids of the types up in the hierarchy
        """
        return self.hierarchy[type_id]
