#!/usr/bin/env python3

import argparse
import collections
import json
import math
import os
import matplotlib.pyplot as plt


COLORS = {
    'PSOE': 'red',
    'PODEMOS': 'mediumpurple',
    'PODEMOS-IU': 'darkviolet',
    'ECP-GUANYEM EL CANVI': 'slateblue',
    'PODEMOS-EU': 'darkmagenta',
    'IU': 'darkred',
    'EH Bildu': 'green',
	  'ERC-SOBIRANISTES': 'yellow',
    'JxCAT-JUNTS': 'turquoise',
    'CUP-PR': 'yellow',
    'EAJ-PNV': 'darkgreen',
    'BNG': 'lightsteelblue',
    'MÁS PAÍS-EQUO': 'limegreen',
    'MÉS COMPROMÍS': 'orange',
    '¡TERUEL EXISTE!': 'forestgreen',
    'PRC': 'yellowgreen',
    'MÉS-ESQUERRA': 'yellow',
    'PUM+J': 'grey',
    'M PAÍS-CHA-EQUO': 'black',
    'PACMA': 'lightgreen',
    'RECORTES CERO-GV': 'black',
    'CCa-PNC-NC': 'gold',
    'FORO': 'royalblue',
    'NA+': 'maroon',
    'Cs': 'darkorange',
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


def dhondt_allocation(votes, seats):
    quotients = dict(votes.items())
    allocations = collections.defaultdict(int)
    allocated = 0

    while allocated < seats:
        winner = max(quotients, key=quotients.get)
        allocations[winner] += 1
        quotients[winner] = votes[winner] / (allocations[winner] + 1)
        allocated += 1

    return allocations


def proportional_allocation(votes, seats):
    allocations = collections.defaultdict(int)
    allocated = 0
    total_votes = sum(votes.values())

    for party, result in sorted(votes.items(),
                                key=lambda item: item[1], reverse=True):
        allocation = max(math.floor(result / total_votes * seats), 1)
        allocations[party] = allocation
        allocated += allocation
        if allocated >= seats:
            break

    return allocations


def do_allocation(proportional, votes, seats):
    if proportional:
        return proportional_allocation(votes, seats)
    return dhondt_allocation(votes, seats)


def plot_allocations(totals, filename, seats):
    label = []
    val = []
    colors = []
    legend = []

    for party, color in COLORS.items():
        if party in totals:
            legend.append('{}:  {}'.format(party, totals[party]))
            if totals[party] >= 10:
                label.append(party)
            else:
                label.append('')
            val.append(totals[party])
            colors.append(color)

    label.append('')
    val.append(sum(val))
    colors.append('white')

    fig = plt.figure(figsize=(12,6), dpi=100)
    ax = fig.add_subplot(1,1,1)
    wedges, _  = ax.pie(val,
                        wedgeprops={'width': 0.4, 'edgecolor': 'w'},
                        labels=label,
                        colors=colors,
                        counterclock=False,
                        startangle=180)
    wedges[-1].set_visible(False)
    ax.legend(legend, loc='lower center', ncol=2)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--proportional', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    seats, barrier, results = read_results(args.input)
    summary, removed = tally(results, barrier)

    totals = collections.defaultdict(int)
    total_removed = 0
    for circunscription, parties in summary.items():
        print('\nCircunscription:', circunscription)
        print('Votes under threshold:', removed[circunscription])
        total_removed += removed[circunscription]

        allocations = do_allocation(
            args.proportional, parties, int(seats[circunscription]))
        print('Allocations')
        for party, allocation in allocations.items():
            print('\t', party, allocation)
            totals[party] += allocation

    print('\nAggregate results')
    for party, allocation in totals.items():
        print('\t', party, allocation)
    total_seats = sum(totals.values())
    print('Total allocated seat:', total_seats)
    print('Total votes under threshold:', total_removed)


    output_name =  '-proportional.png' if args.proportional else '-dhondt.png'
    plot_allocations(
        totals, os.path.splitext(args.input)[0] + output_name, total_seats)
