import os
import sys
import vcf
import h5py
import argparse
import numpy as np
import logging
import random
from collections import defaultdict
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.preprocessing import LabelBinarizer
from pprint import pprint
from os import listdir, makedirs, path

from Bio import Seq, SeqIO

from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Input, Dense, Dropout, SpatialDropout2D, Flatten, Reshape, merge
from keras.layers.convolutional import Conv1D, Convolution2D, MaxPooling1D, MaxPooling2D

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("indel")
logger.setLevel(logging.INFO)

BASES = ['A','C','G','T']
dna_encoder = LabelBinarizer(sparse_output=True).fit(BASES)

def load_fasta_reference(in_path):
    contigs = {}
    logger.info("Reading fasata file {}".format(in_path))
    start = timer()
    fasta = SeqIO.to_dict(SeqIO.parse(in_path, "fasta"))
    logger.info("Reading fasata file loaded in {}.".format(timer() - start))
    for chrom, sequence in fasta.iteritems():
        start = timer()
        logger.info("Encoding chromosome {}".format(chrom))
        one_hot = dna_encoder.transform(list(str(sequence.seq)))
        contigs[chrom] = one_hot.astype(np.int8).todense()
        logger.info("Chrom {} encoded in {}.".format(chrom, timer() - start))
    return contigs

def write_bitpacked_reference(reference, out_path):
    logger.info("Writing reference to {}".format(out_path))
    if not path.exists(out_path):
        makedirs(out_path)
    for contig, sequence in reference.iteritems():
        np.save(out_path + "/{}.npy".format(contig),np.packbits(sequence))

def load_bitpacked_reference(in_path):
    logger.info("Loading bitpacked reference from {}".format(in_path))
    contigs = {}
    for f in listdir(in_path):
        if f.endswith(".npy"):
            packed = np.load(in_path + "/" + f)
            contigs[f[:-4]] = np.unpackbits(packed).reshape(packed.shape[1]*2,4)
    return contigs

def main(args):
    if args.ref_fasta:
        reference = load_fasta_reference(args.ref_fasta)
        write_bitpacked_reference(args.ref_fasta + ".bp")
    else:
        logger.info("Loading reference pickle")
        reference = load_bitpacked_reference(args.ref)

    logger.info("Loading training data")
    training, labels = vcf_to_indel_tensors(args.vcf, args.max_training, args.window_size, args.neg_train_window, reference)
    logger.info("Loaded training examples with dimensions {}.".format(str(training.shape)))

    logger.info("Splitting training data into training and testing sets.")
    train, test = split_data([training,labels], [0.8,0.2])

    logger.info("Training shape: {}, labels shape: {}".format(train[0].shape, train[1].shape))

    logger.info("Creating model")
    model = make_indel_model(args.window_size)

    logger.info("Training model.")
    history = model.fit(train[0], train[1], batch_size=32, validation_split=0.2, shuffle=True, epochs=10, callbacks=get_callbacks(args.output+".callbacks"))

    logger.info("Evaluating model.")
    eval_values = model.evaluate(test[0], test[1])
    pprint(", ".join(["{}: {}".format(metric,value) for metric, value in zip(model.metrics_names, eval_values)]))

    logger.info("Plotting metrics.")
    plot_metric_history(history, args.output + ".metrics_history.pdf", "Indel model")


def vcf_to_indel_tensors(vcf_path, max_training, window_size, neg_train_window, reference, squash_multi_allelic = False):
    indels = []
    positive_training_pos = defaultdict(dict)

    vcf_reader = vcf.Reader(open(vcf_path, 'r'))

    # Get positive training examples
    start = timer()
    for variant in vcf_reader:
        if max_training >= 0 and len(indels) >= max_training:
            break

        if not variant.is_snp and not (squash_multi_allelic and variant.POS in positive_training_pos[variant.CHROM]):
            pos = variant.POS
            indels.append(reference[variant.CHROM][pos - window_size: pos + window_size + 1])
            while pos in positive_training_pos[variant.CHROM]:
                pos+=1
            positive_training_pos[variant.CHROM][pos] = 1

    logger.info("Positive training examples loaded in {}.".format(timer() - start))

    # Get negative training examples
    start = timer()
    negative_training = []
    for chrom, positions in positive_training_pos.iteritems():
        for pos in get_negative_positions(positions, neg_train_window):
            negative_training.append(reference[chrom][pos - window_size: pos + window_size + 1])

    logger.info("Negative training examples loaded in {}.".format(timer() - start))
    # Create labels
    labels = np.concatenate(
        (
            np.full((len(indels), 2), np.array([0, 1])),  # positive labels
            np.full((len(negative_training), 2), np.array([1, 0]))  # negative labels
        ), axis=0)

    return np.asarray(indels + negative_training), labels

def one_hot_encode(data, encoder):
    all_data = "".join(data)
    res = encoder.transform(list(all_data))
    return np.asarray(np.split(res, len(data)))


def get_training_tensor(chrom, pos, reference, window_size, encoder):
    contig = reference[chrom]
    record = contig[pos - window_size: pos + window_size + 1]
    return encoder.transform(list(record))


def get_negative_positions(positions, neg_train_window):
    for pos in positions.keys():
        neg_pos = random.randint(pos - neg_train_window, pos + neg_train_window)
        while neg_pos in positions:
            neg_pos = random.randint(pos - neg_train_window, pos + neg_train_window)
        yield neg_pos


def make_indel_model(window_size):
    indel = Input(shape=(2 * window_size + 1, len(BASES)), name="indel")
    x = Conv1D(filters=200, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(indel)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=200, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(x)
    x = Dropout(0.2)(x)
    x = Conv1D(filters=100, kernel_size=12, activation="relu", kernel_initializer='glorot_normal')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(units=64, kernel_initializer='normal', activation='relu')(x)
    prob_output = Dense(units=2, kernel_initializer='normal', activation='softmax')(x)

    model = Model(inputs=[indel], outputs=[prob_output])

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
    adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

    model.compile(optimizer=adamo, loss='binary_crossentropy', metrics=['accuracy'])

    logger.info('model summary:\n{}'.format(model.summary()))

    return model


def bed_file_to_dict(bed_file):
    ''' Create a dict to store intervals from a bed file.

    Arguments:
        bed_file: the file to load

    Returns:
        bed: dict where keys in the dict are contig ids
            values are a tuple of arrays the first array
            in the tuple contains the start positions
            the second array contains the end positions.
    '''
    bed = {}

    with open(bed_file)as f:
        for line in f:
            parts = line.split()
            contig = parts[0]
            lower = int(parts[1])
            upper = int(parts[2])

            if contig not in bed.keys():
                bed[contig] = ([], [])

            bed[contig][0].append(lower)
            bed[contig][1].append(upper)

    for k in bed.keys():
        bed[k] = (np.array(bed[k][0]), np.array(bed[k][1]))

    return bed


def in_bed_file(bed_dict, contig, pos):
    # Exclusive
    lows = bed_dict[contig][0]
    ups = bed_dict[contig][1]
    return np.any((lows < pos) & (pos < ups))


def split_data(datasets, subset_ratios, sequential=False):
    """

    Splits datasets in len(subset_ratios) subsets with ratios as specified in subset_ratios.
    If ratios don't sum up to 1, they are normalized first.

    :param list of array datasets: input datasets
    :param list of float subset_ratios: list of subsets with their respective ratios
    :param bool sequential: If set, the data is partitioned sequentially rather than shuffled
    :return: Subsets
    :rtype: list of array
    """

    samples = datasets[0].shape[0]
    sum_ratios = sum(subset_ratios)
    results_length = [int(x / sum_ratios * samples) for x in subset_ratios]

    indices = range(samples)

    if not sequential:
        random.shuffle(indices)

    results = []
    i = 0
    for n in results_length:
        results.append([x[indices[i:n+i]] for x in datasets])
        i += n

    return results


def plot_metric_history(history, output, title):
    """
    Plots metrics history as pdf.

    :param history:
    :param str output: output file name
    :param str title: plot title
    :return:
    """
    # list all data in history
    print(history.history.keys())

    row = 0
    col = 0
    num_plots = len(history.history) / 2.0  # valid and train plot together
    rows = 4
    cols = max(2, int(math.ceil(num_plots / float(rows))))

    f, axes = plt.subplots(rows, cols, sharex=True, figsize=(36, 24))
    for k in history.history.keys():

        if 'val' not in k:

            axes[row, col].plot(history.history[k])
            axes[row, col].plot(history.history['val_' + k])

            axes[row, col].set_ylabel(str(k))
            axes[row, col].legend(['train', 'valid'], loc='upper left')
            axes[row, col].set_xlabel('epoch')

            row += 1
            if row == rows:
                row = 0
                col += 1
                if row * col >= rows * cols:
                    break

    axes[0, 1].set_title(title)
    plt.savefig(output)


def get_callbacks(output_prefix, patience=2, batch_size=32):
    checkpointer = ModelCheckpoint(filepath=output_prefix + ".callbacks", verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)
    tensorboard = TensorBoard(log_dir=output_prefix + "/tensorboard", histogram_freq=1, batch_size=batch_size)
    return [checkpointer, earlystopper, tensorboard]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', help='One-hot-enoded reference pickle (required if not using --ref_fasta)', required=False)
    parser.add_argument('--ref_fasta', help='Reference fasta. If used, the reference will be one-hot-encoded and saved as a npy files. (required if not using --ref)', required=False)
    parser.add_argument('--vcf', help='VCF containing indel training examples', required=True)
    parser.add_argument('--output', help='Output prefix', required=True)
    parser.add_argument('--window_size', help='Size of the window to consider on each side of the indel (default 30).',
                        required=False, default=30, type=int)
    parser.add_argument('--max_training',
                        help='Maximum number of training examples to use for each class (default 100,000). Set to -1 to use all.',
                        required=False, default=100000, type=int)
    parser.add_argument('--neg_train_window',
                        help='Size of the window to consider on each side of the indel to sample a negative training example (default 1000).',
                        required=False, default=1000, type=int)
    args = parser.parse_args()

    if int(args.ref_fasta is not None) + int(args.ref is not None) != 1:
        sys.exit("One and only one of --ref_fasta or --ref is required.")

    main(args)
