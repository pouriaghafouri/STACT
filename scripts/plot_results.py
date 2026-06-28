#!/usr/bin/env python3
import csv, sys
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
rows = list(csv.DictReader(open(path)))
labels = [r['engine'] + '/' + r['layout'] for r in rows]
ms = [float(r['avg_ms']) for r in rows]
thr = [float(r['throughput_mvalues_s']) for r in rows]

plt.figure()
plt.bar(labels, ms)
plt.ylabel('Average ms')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('avg_ms.png')

plt.figure()
plt.bar(labels, thr)
plt.ylabel('Mvalues/s')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('throughput.png')
print('Wrote avg_ms.png and throughput.png')
