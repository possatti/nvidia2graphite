#!/usr/bin/env python3
"""Send metrics from NVIDIA GPUs to Graphite

Uses nvidia-smi to collect metrics about NVIDIA GPUs and sends the data to
graphite. The configuration is done in nvidia2graphite.conf which is located
in /etc/ after installation. The server as well as the sent metrics are defined
there.

Installation command (as superuser):
    $ python3 setup.py install

Note: Uses Python3 but may work with python2 too (not tested).

Requirements: graphitesend

License: MIT

Author: Stefan Kroboth <stefan.kroboth@uniklinik-freiburg.de>
"""

from __future__ import print_function
import subprocess
import configparser
from xml.etree import ElementTree
import re
import sys
import time
import graphitesend
import argparse

# pylint: disable=invalid-name


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='/etc/nvidia2graphite.conf')
    parser.add_argument('-n', '--dryrun', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Read in the config file
    conf = configparser.ConfigParser()
    conf.read(args.config)
    if args.debug:
        print('DEBUG: conf={}'.format(conf), file=sys.stderr)

    # Extract metrics
    metrics = []
    for key in conf['Metrics']:
        metrics.append(conf['Metrics'][key])

    # Extract graphite server information
    graphite_server = conf['Graphite']['host']
    graphite_port = int(conf['Graphite']['port'])
    wait_time = float(conf['Graphite']['interval'])

    g = None
    while 1:
        data = subprocess.check_output(['nvidia-smi', '-q', '-x'])
        root = ElementTree.fromstring(data)

        # Loop over GPUs
        gpu_id = 1
        for gpu in root:
            # ignore non-gpu tags
            if gpu.tag != 'gpu':
                continue

            # parse XML data and compile dictionary
            metric_dict = dict()
            for metric in metrics:
                curr_level = gpu
                for level in metric.split('.'):
                    curr_level = curr_level.find(level)
                data = re.search(r"(\d+\.?\d*)", curr_level.text).group(1)
                if data is not None and data != '':
                    metric_dict[metric] = data

            # setup graphite
            if g is None:
                try:
                    g = graphitesend.init(prefix=conf['Graphite']['prefix'],
                                          group='gpu' + str(gpu_id),
                                          graphite_server=graphite_server,
                                          graphite_port=graphite_port,
                                          dryrun=args.dryrun)
                except graphitesend.graphitesend.GraphiteSendException as e:
                    print(repr(e), file=sys.stderr)
                    print('Will retry again in {} seconds.'.format(wait_time), file=sys.stderr)

            # send dictionary
            if g is not None:
                sent = g.send_dict(metric_dict)
                if args.debug:
                    print('DEBUG: sent={}'.format(sent), file=sys.stderr)

            gpu_id += 1

        time.sleep(wait_time)

if __name__ == '__main__':
    main()
