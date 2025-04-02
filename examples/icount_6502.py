import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="ASM file")
args = parser.parse_args()

m = {}
tot=0
for s in open(args.filename):
    s = s[16:19].strip()
    if len(s) == 3 and s.isalpha() and s.isupper():
        if s not in m:
            m[s] = 0
        m[s] += 1
        tot+=1
r = sorted(m.keys(), key=lambda s: m[s])
for k in r:
    print(k, m[k])
print('   ',tot)
