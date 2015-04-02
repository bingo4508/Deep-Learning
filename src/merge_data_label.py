import sys

data = sys.argv[1]
label = sys.argv[2]
out = sys.argv[3]
label_i = 0
label_map = {}
label_set = set()

with open(label, 'r') as f:
    for l in f:
        l = l.strip().split(',')
        label_set.add(l[1])
        label_map[l[0]] = l[1]

label_set = {e:str(i) for i, e in enumerate(list(label_set))}
for k in label_map:
    label_map[k] = label_set[label_map[k]]

with open(data, 'r') as f:
    with open(out, 'w') as fo:
        for l in f:
            l = l.strip().split(' ')
            l.append(label_map[l[0]])
            fo.write("%s\n" % ' '.join(l))
