#!/usr/bin/env python3

import argparse
import collections
import json
import math
import matplotlib.pyplot as plt


COLORS = {
    'PSOE': 'red',
    'PODEMOS': 'purple',
    'PODEMOS-IU': 'purple',
    'ECP-GUANYEM EL CANVI': 'purple',
    'PODEMOS-EU': 'purple',
    'IU': 'red',
    'EH Bildu': 'green',
	  'ERC-SOBIRANISTES': 'yellow',
    'JxCAT-JUNTS': 'blue',
    'CUP-PR': 'yellow',
    'EAJ-PNV': 'green',
    'BNG': 'green',
    'MÁS PAÍS-EQUO': 'green',
    'MÉS COMPROMÍS': 'orange',
    '¡TERUEL EXISTE!': 'green',
    'PRC': 'green',
    'CCa-PNC-NC': 'blue',
    'FORO': 'blue',
    'NA+': 'red',
    'Cs': 'orange',
	  'VOX': 'green',
    'PP': 'blue',
}


def read_results(filename):
    with open(filename, 'r') as f:
        results = json.load(f)

        seats = results['Seats']
        del results['Seats']

        barrier = results['Barrier']
        del results['Barrier']

        return seats, barrier, results


def tally(results, barrier):
    summary = collections.defaultdict(lambda: collections.defaultdict(int))
    totals = collections.defaultdict(int)
    removed = collections.defaultdict(int)

    for party, result in results.items():
        for circunscription, votes in result.items():
            summary[circunscription][party] += int(votes)
            totals[circunscription] += int(votes)

    for circunscription, parties in summary.items():
        threshold = math.floor(totals[circunscription]
                               * float(barrier[circunscription]))
        for party, votes in parties.items():
            if votes < threshold:
                parties[party] = 0
                removed[circunscription] += votes

    return summary, removed


def dhondt(votes, seats):
    quotients = dict(votes.items())
    allocations = collections.defaultdict(int)
    allocated = 0

    while allocated < seats:
        winner = max(quotients, key=quotients.get)
        allocations[winner] += 1
        quotients[winner] = votes[winner] / (allocations[winner] + 1)
        allocated += 1

    return allocations


def plot_allocations(totals):
    label = []
    val = []
    colors = []

    for party, color in COLORS.items():
        if party in totals:
            if totals[party] >= 10:
                label.append(party)
            else:
                label.append('')
            val.append(totals[party])
            colors.append(color)

    label.append('')
    val.append(sum(val))
    colors.append('white')

    def show_value(pct):
        v = int(pct / 100 * sum(val))
        if 350 > v >= 10:
            return str(v)
        return ''

    fig = plt.figure(figsize=(12,6), dpi=100)
    ax = fig.add_subplot(1,1,1)
    wedges, _, _ = ax.pie(val,
                          wedgeprops={'width': 0.4, 'edgecolor': 'w'},
                          labels=label,
                          colors=colors,
                          autopct=show_value,
                          counterclock=False,
                          startangle=180)
    wedges[-1].set_visible(False)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()

    seats, barrier, results = read_results(args.input)
    summary, removed = tally(results, barrier)

    totals = collections.defaultdict(int)
    for circunscription, parties in summary.items():
        print('\nCircunscription:', circunscription)
        print('Votes under threshold:', removed[circunscription])

        allocations = dhondt(parties, int(seats[circunscription]))
        print('Allocations')
        for party, allocation in allocations.items():
            print('\t', party, allocation)
            totals[party] += allocation

    print('\nAggregate results')
    for party, allocation in totals.items():
        print('\t', party, allocation)

    plot_allocations(totals)
