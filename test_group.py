import group_adjust

average = 0

for i in range(10):
    average += group_adjust.test_performance()

average /= 10

with open('output.txt', 'w+') as f:
    f.write("Average computation time over 10 seconds is: " + str(average))
