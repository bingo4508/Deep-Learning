import sys

# Load map
def load_map(fname):
    m = {}
    with open(fname, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            m[l[0]] = l[2]
    return m


state_result = sys.argv[1]    # ',' delimited
final_result = sys.argv[2]
state_map = sys.argv[3]     # tab delimited


state_map = load_map(state_map)

with open(state_result, 'r') as f:
    with open(final_result, 'w') as fo:
        fo.write('Id,Prediction\n')
        for l in f:
            l = l.strip().split(',')
            fo.write('%s,%s\n' % (l[0],state_map[l[1]]))
