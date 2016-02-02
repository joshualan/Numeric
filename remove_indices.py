def remove_indices(lst, index):
    return [l for i,l in enumerate(lst) if i not in index]

def remove_indices2(lst, index):
    
    # Explicitly instantiate the dictionary
    length = len(lst)
    d = {x: [] for x in range(length)}
    
    # Build up the dict of the indices we want to remove
    # per list.
    # Ex: If we want to remove elements 0 and 2 in list 0
    # d[0] = [0,2]
    result = []
    for x, y in index:
        d[x].append(y)
        
    # Remove the elements who's indices are in the dict
    for k,v in d.items():
        tmp = remove_indices(lst[k], v)
        result.append(tmp)
        
    return result
