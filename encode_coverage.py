import gzip
import numpy as np
import glob
import argparse
from os import makedirs, path
import pandas as pd
import csv
import sys

def main(args):

    contigs = {}
    max_coverage = 0
    for input in args.input:
        for file in glob.glob(input):
            print "Loading coverage from file: {}".format(file)
            df = pd.read_csv(file, sep="\t", usecols=['#chrom','pos','mean'], dtype={'#chrom': str, 'pos': np.int32, 'mean': np.float32})
            if df['#chrom'].iloc[0] != df['#chrom'].iloc[-1]:
                sys.exit("ERROR: This script expects coverage for a single contig per file.")
            coverage = np.zeros(df['pos'].iloc[-1], dtype=np.float32)
            coverage[df['pos'].values - 1] = df['mean'].values
            contigs.update({df['#chrom'].iloc[0] : coverage})
            max_coverage = max(max_coverage, np.max(df['mean'].values))

    # print "Normalizing coverage using {} points to compute the norm".format(args.n_points_for_norm)
    # for contig, coverage in contigs.iteritems():
    #     print "Max {} coverage: {}".format(contig, np.max(coverage))
    #
    # total_length = float(sum([len(x) for x in contigs.values()]))
    # print total_length
    # p_per_base = args.n_points_for_norm / total_length
    # coverage_sample = np.asarray([np.random.choice(x, int(p_per_base * len(x))) for x in contigs.values()])
    #max = np.max(coverage_sample)
    #norm = np.linalg.norm(np.asarray([ np.random.choice(x, int(p_per_base * len(x))) for x in contigs.values() ]).flatten())

    print "Max coverage value found: {}".format(max_coverage)

    print "Writing coverage"
    if not path.exists(args.output):
        makedirs(args.output)

    for contig, coverage in contigs.iteritems():
        np.save(args.output + "/{}.npy".format(contig), (coverage / max_coverage).astype(np.float16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Coverage file(s).", nargs='+')
    parser.add_argument("--output", help="Coverage output path.")
    parser.add_argument("--n_points_for_norm", help="Number of coverage points to use for normalization. (default 50M)", type=int, default=50000000)

    main(parser.parse_args())




