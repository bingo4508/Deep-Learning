import sys

data = sys.argv[1]
label = sys.argv[2]
out_data = sys.argv[3]
if len(sys.argv) == 5:
    out_map = sys.argv[4]
else:
    out_map = None

label_i = 0
label_map = {}
label_set = set()

with open(label, 'r') as f:
    for l in f:
        l = l.strip().split(',')
        label_set.add(l[1])
        label_map[l[0]] = l[1]

if out_map is not None:
    label_set = {e:str(i) for i, e in enumerate(list(label_set))}
    for k in label_map:
        label_map[k] = label_set[label_map[k]]

# Output merged data
with open(data, 'r') as f:
    with open(out_data, 'w') as fo:
        for l in f:
            l = l.strip().split(' ')
            l.append(label_map[l[0]])
            fo.write("%s\n" % ' '.join(l))

# Output map
if out_map is not None:
    with open(out_map, 'w') as f:
        for k,v in label_set.items():
            # ex: 10    fil
            f.write('%s\t%s\n' % (v,k))
