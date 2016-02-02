import remove_indices as rm

l1 = [['print', 'hello', 'world'], ['luke','sky', 'walker'],
      ['hammer', 'time']]
i1 = [(0,0), (0,2), (1,1)]

l = rm.remove_indices(l1[1], [0,2])
assert l == ['sky']

l = rm.remove_indices2(l1, i1)
assert l == [['hello'], ['luke', 'walker'], ['hammer', 'time']]

