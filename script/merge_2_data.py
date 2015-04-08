import sys

data1 = sys.argv[1]
data2 = sys.argv[2]
output = sys.argv[3]

with open(output, 'w') as fo:
    with open(data1, 'r') as f1:
        with open(data2, 'r') as f2:
            while True:
                line1 = f1.readline()
                if not line1:
                    break
                line2 = f2.readline()
                line1 = line1.strip().split()
                line2 = line2.strip().split()
                fo.write(' '.join([line1[0]]+line1[1:]+line2[1:])+'\n')
