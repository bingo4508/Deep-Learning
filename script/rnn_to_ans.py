import sys

input = sys.argv[1]
output = sys.argv[2]

m = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e'}

with open(output, 'w') as fo:
    fo.write("Id,Answer\n")
    with open(input, 'r') as f:
        for i,l in enumerate(f):
            fo.write("%d,%s\n" % (i+1, m[int(l.strip())]))
