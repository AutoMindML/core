def get_sublist_before_target(lst, target):
    if target in lst:
        index = lst.index(target)
        return lst[:index]
    else:
        return []
