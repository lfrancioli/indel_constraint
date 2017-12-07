import argparse
import scipy.io as spio
import sys
import pickle


def main(args):

    mat = spio.loadmat(args.input)

    if args.list_keys:
        print "Keys:"
        print "\n".join(mat.keys())
    else:
        res={}
        for i in range(0, len(mat[args.key])):
            chrom = str(i+1)
            if i == 22:
                chrom = "X"
            elif i == 23:
                chrom = "Y"

            res[chrom] = mat[args.key][i][0]

        pickle.dump(res, open(args.output,'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input Matlab file.')
    parser.add_argument('--output', help='Output pickle.', required=False)
    parser.add_argument('--key', help='Matlab dict key containing replication timing data.', required=False)
    parser.add_argument('--list_keys', help='Lists Matlab dict keys only.', required=False, action='store_true')
    args = parser.parse_args()

    if not args.list_keys and not (args.output and args.key):
            sys.exit("--input, --output and --key required unless --list_keys is provided")

    main(args)

