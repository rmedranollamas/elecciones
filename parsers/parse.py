#!/usr/bin/env python3

import collections
import csv
import json


def read_csv(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        return list(reader)


def party_to_index(parties):
    return {p.strip(): i for i, p in enumerate(parties) if p}


def filter_lines(lines):
    idx = party_to_index(lines[0])
    del lines[0]
    del lines[0]
    del lines[-1]
    del lines[-1]
    return idx, lines


def gen_summary(parties, results):
    summary = collections.defaultdict(lambda: collections.defaultdict(int))
    for result  in results:
        province = result[0].strip()
        summary['Barrier'][province] = 0.03
        summary['Seats'][province] = -1
        for party, idx in parties.items():
            votes = int(result[idx].replace(',', ''))
            if votes > 0:
                summary[party][province] = votes
    return summary


def dump_to_json(summary):
    with open('nov2019.json', 'w') as f:
        json.dump(summary, f, indent=4)


if __name__ == '__main__':
    summary = gen_summary(*filter_lines(read_csv('nov2019.csv')))
    dump_to_json(summary)
