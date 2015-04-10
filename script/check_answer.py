import sys


predict = sys.argv[1]
answer = sys.argv[2]

m = {}
with open(answer, 'r') as f:
    for l in f:
        l=l.strip().split(',')
        m[l[0]] = l[1]

with open(predict, 'r') as f:
    f.readline()
    ln = 0
    correct = 0
    for l in f:
        ln += 1
        l=l.strip().split(',')
        if l[1] == m[l[0]]:
            correct += 1
    print("Accuracy: %f" % (float(correct)/ln))
        
