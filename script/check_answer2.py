import sys


predict = sys.argv[1]
answer = sys.argv[2]

with open(answer, 'r') as f:
    l1 = [l.strip() for l in f ]

with open(predict, 'r') as f:
    l2 = [l.strip() for l in f]

correct = 0
for m, n in zip(l1, l2):
    if m == n:
	correct += 1

print "Accuracy: %f" % (float(correct)/len(l1))
        
