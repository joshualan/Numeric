import pytest
from datetime import datetime
import numpy as np

# Your task is to write the group adjustment method below. There are some
# unimplemented unit_tests at the bottom which also need implementation.
# Your solution can be pure python, pure NumPy, pure Pandas
# or any combination of the three.  There are multiple ways of solving this
# problem, be creative, use comments to explain your code.

# Group Adjust Method
# The algorithm needs to do the following:
# 1.) For each group-list provided, calculate the means of the values for each
# unique group.
#
#   For example:
#   vals       = [  1  ,   2  ,   3  ]
#   ctry_grp   = ['USA', 'USA', 'USA']
#   state_grp  = ['MA' , 'MA' ,  'CT' ]
#
#   There is only 1 country in the ctry_grp list.  So to get the means:
#     USA_mean == mean(vals) == 2
#     ctry_means = [2, 2, 2]
#   There are 2 states, so to get the means for each state:
#     MA_mean == mean(vals[0], vals[1]) == 1.5
#     CT_mean == mean(vals[2]) == 3
#     state_means = [1.5, 1.5, 3]
#
# 2.) Using the weights, calculate a weighted average of those group means
#   Continuing from our example:
#   weights = [.35, .65]
#   35% weighted on country, 65% weighted on state
#   ctry_means  = [2  , 2  , 2]
#   state_means = [1.5, 1.5, 3]
#   weighted_means = [2*.35 + .65*1.5, 2*.35 + .65*1.5, 2*.35 + .65*3]
#
# 3.) Subtract the weighted average group means from each original value
#   Continuing from our example:
#   val[0] = 1
#   ctry[0] = 'USA' --> 'USA' mean == 2, ctry weight = .35
#   state[0] = 'MA' --> 'MA'  mean == 1.5, state weight = .65
#   weighted_mean = 2*.35 + .65*1.5 = 1.675
#   demeaned = 1 - 1.675 = -0.675
#   Do this for all values in the original list.
#
# 4.) Return the demeaned values

# Hint: See the test cases below for how the calculation should work.


def group_adjust(vals, groups, weights):
    """
    Calculate a group adjustment (demean).
    Parameters
    ----------
    vals    : List of floats/ints
        The original values to adjust
    groups  : List of Lists
        A list of groups. Each group will be a list of strings
    weights : List of floats
        A list of weights for the groupings.
    Returns
    -------
    A list-like demeaned version of the input values
    """

    # Check to make sure inputs are valid before doing any processing as
    # to not waste cycles doing computation
    vals_len = len(vals)

    if not all(len(group) == vals_len for group in groups):
        raise ValueError('All groups need to be the same length as vals')

    if len(groups) != len(weights):
        raise ValueError('Each group-list needs its own weight')

    group_means = []

    # Build the means for each group
    for group in groups:
        group_average = {}
        name_count = {}
        for i, name in enumerate(group):

            # Encountering a None means we essentially
            # cut this value out of all the calculations
            if vals[i] is None:
                continue

            # This sums up the values for each unique group
            if name in group_average:
                group_average[name] += vals[i]
            else:
                # Ensures that the value is a float
                group_average[name] = float(vals[i])

            # This keeps track of how many instances there are of each group
            if name in name_count:
                name_count[name] += 1
            else:
                name_count[name] = 1

        # Here is where we finally get the average of each group
        group_average = {k: v / name_count[k]
                 for k, v in group_average.items()}

        # Old, slower computation
        # for k, v in group_average.items():
        #     group_average[k] = v / name_count[k]

        # We create the equivalent mean list for each group-list
        group_mean = [group_average[name] for name in group]

        # Old slower computation
        # group_mean = []
        # for name in group:
        #     group_mean.append(group_average[name])

        # Incrementally create a matrix of mean lists
        group_means.append(group_mean)

    # The dot product between two 2D arrays is just matrix multiplication.
    # This will yield the weighted means.
    weighted_means = np.dot([weights], group_means)

    # Return a list of demeaned vals by subtracting the calculated weighted
    # mean from each element in vals
    demeaned_vals = []
    for i, val in enumerate(vals):
        # Once again, a None is a no-op
        if val is None:
            demeaned_vals.append(val)
        else:
            demeaned_vals.append(val - weighted_means[0][i])

    # List comprehension is slightly faster but affects readability
    # demeaned_vals = [None if val is None else val - weighted_means[0][i]
    #                 for i, val in enumerate(vals)]

    return demeaned_vals

def test_three_groups():
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'MA', 'MA', 'RI', 'RI']
    grps_3 = ['WEYMOUTH', 'BOSTON', 'BOSTON', 'PROVIDENCE', 'PROVIDENCE']
    weights = [.15, .35, .5]

    adj_vals = group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    # 1 - (USA_mean*.15 + MA_mean * .35 + WEYMOUTH_mean * .5)
    # 2 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # 3 - (USA_mean*.15 + MA_mean * .35 + BOSTON_mean * .5)
    # etc ...
    # Plug in the numbers ...
    # 1 - (.15*2 + .35*2 + .5*1)   # -0.5
    # 2 - (.15*2 + .35*2 + .5*2.5) # -.25
    # 3 - (.15*2 + .35*2 + .5*2.5) # 0.75
    # etc...

    answer = [-0.770, -0.520, 0.480, 1.905, -1.095]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5


def test_two_groups():
    vals = [1, 2, 3, 8, 5]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)
    # 1 - (.65*2 + .35*1)   # -0.65
    # 2 - (.65*2 + .35*2.5) # -.175
    # 3 - (.65*2 + .35*2.5) # -.825
    answer = [-1.81999, -1.16999, -1.33666, 3.66333, 0.66333]
    for ans, res in zip(answer, adj_vals):
        assert abs(ans - res) < 1e-5


def test_missing_vals():
    # If you're using NumPy or Pandas, use np.NaN
    # If you're writing python, use None
    # vals = [1, np.NaN, 3, 5, 8, 7]
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65, .35]

    adj_vals = group_adjust(vals, [grps_1, grps_2], weights)

    # This should be None or np.NaN depending on your implementation
    # please feel free to change this line to match yours
    # answer = [-2.47, np.NaN, -1.170, -0.4533333, 2.54666666, 1.54666666]
    answer = [-2.47, None, -1.170, -0.4533333, 2.54666666, 1.54666666]

    for ans, res in zip(answer, adj_vals):
        if ans is None:
            assert res is None
        elif np.isnan(ans):
            assert np.isnan(res)
        else:
            assert abs(ans - res) < 1e-5


def test_weights_len_equals_group_len():
    # Need to have 1 weight for each group

    # vals = [1, np.NaN, 3, 5, 8, 7]
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA', 'USA', 'USA', 'USA', 'USA', 'USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65]

    with pytest.raises(ValueError) as excinfo:
        group_adjust(vals, [grps_1, grps_2], weights)
    assert 'Each group-list needs its own weight' in str(excinfo.value)

def test_group_len_equals_vals_len():
    # The groups need to be same shape as vals
    vals = [1, None, 3, 5, 8, 7]
    grps_1 = ['USA']
    grps_2 = ['MA', 'RI', 'RI', 'CT', 'CT', 'CT']
    weights = [.65]

    with pytest.raises(ValueError) as excinfo:
        group_adjust(vals, [grps_1, grps_2], weights)
    assert 'All groups need to be the same length as vals' in str(excinfo.value)

def test_performance():
    vals = 1000000*[1, None, 3, 5, 8, 7]
    # If you're doing numpy, use the np.NaN instead
    # vals = 1000000 * [1, np.NaN, 3, 5, 8, 7]
    grps_1 = 1000000 * [1, 1, 1, 1, 1, 1]
    grps_2 = 1000000 * [1, 1, 1, 1, 2, 2]
    grps_3 = 1000000 * [1, 2, 2, 3, 4, 5]
    weights = [.20, .30, .50]

    start = datetime.now()
    group_adjust(vals, [grps_1, grps_2, grps_3], weights)
    end = datetime.now()
    diff = end - start
    print 'Total performance test time: {}'.format(diff.total_seconds())

    # Added a return so I can test this
    return diff.total_seconds()
