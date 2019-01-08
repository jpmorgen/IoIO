#!/usr/bin/python3
import csv

# See comments in read_ap
from read_ap import ADU2R_adjust

line = 'Na'
ap_sum_fname = '/data/io/IoIO/reduced/ap_sum.csv'
Na_ap_sum_fname = '/data/io/IoIO/reduced/Na_ap_sum.csv'
rlist = []
with open(ap_sum_fname, newline='') as csvin:
    csvr = csv.DictReader(csvin, quoting=csv.QUOTE_NONNUMERIC)
    fieldnames = csvr.fieldnames
    ap_keys = [k for k in fieldnames
               if ('AP' in k
                   or 'On' in k
                   or 'Off' in k)]
    for row in csvr:
        if row['LINE'] != line:
            continue
        for ap in ap_keys:
            # Apply ADU2R adjustment for sodium
            if row['LINE'] == 'Na':
                row[ap] *= ADU2R_adjust
        rlist.append(row)
slist = sorted(rlist, key = lambda k: k['TMID'])
with open(Na_ap_sum_fname, 'w', newline='') as csvout:
    csvdw = csv.DictWriter(csvout, fieldnames=fieldnames,
                           quoting=csv.QUOTE_NONNUMERIC)
    csvdw.writeheader()
    for row in slist:
        csvdw.writerow(row)

#line = 'Na'
#ap_sum_fname = '/data/io/IoIO/reduced/ap_sum.csv'
#Na_ap_sum_fname = '/data/io/IoIO/reduced/Na_ap_sum.csv'
#with open(ap_sum_fname, newline='') as csvin:
#    csvr = csv.DictReader(csvin, quoting=csv.QUOTE_NONNUMERIC)
#    fieldnames = csvr.fieldnames
#    with open(Na_ap_sum_fname, 'w', newline='') as csvout:
#        csvdw = csv.DictWriter(csvout, fieldnames=fieldnames,
#                               quoting=csv.QUOTE_NONNUMERIC)
#        csvdw.writeheader()
#        for row in csvr:
#            if row['LINE'] != line:
#                continue
#            csvdw.writerow(row)
