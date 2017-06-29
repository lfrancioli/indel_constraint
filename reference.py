from Bio import Seq, SeqIO
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import numpy as np
import logging
from timeit import default_timer as timer
from numpy import in1d
from collections import OrderedDict
import scipy.sparse as sp
from os import listdir, makedirs, path

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("BitPackedFasta")
logger.setLevel(logging.INFO)

_BASES = ['A','C','G','T']
_dna_encoder = LabelBinarizer(sparse_output=True).fit(_BASES)




def _encode_sequence(seq):
    ##This is equivalent to above but uses int8 instead of int64 for data;
    # may be useful if memory becomes an issue while loading again
    # _classes = np.array(['A', 'C', 'G', 'T'], dtype='|S1')
    # y = np.ravel(list(seq))
    # # pick out the known labels from y
    # y_in_classes = in1d(y, _classes)
    # y_seen = y[y_in_classes]
    # indices = np.searchsorted(_classes, y_seen).astype(np.int8)
    # indptr = np.hstack((0, np.cumsum(y_in_classes)))
    #
    # data = np.empty_like(indices)
    # data.fill(pos_label)
    # Y = sp.csr_matrix((data, indices, indptr),
    #                   shape=(len(seq), len(_classes)))
    # return BitPackedSequence(np.packbits(Y.todense()))
    one_hot = _dna_encoder.transform(list(str(seq))).astype(np.int8).astype(np.int8)
    return np.packbits(one_hot.todense())

class BitPackedSequence:

    def __init__(self, encoded_seq, length, left_offset):
        self._bitpacked_seq = encoded_seq
        self._length = length
        self._left_offset = left_offset

    def __init__(self, seq):
        self._bitpacked_seq = _encode_sequence(seq)
        self._length = len(seq)
        self._left_offset = 0

    def __len__(self):
        return self._length

    def __str__(self):
        return "".join(_dna_encoder.inverse_transform(self.to_tensor()).flatten())

    def __getitem__(self, item):
        if not (isinstance(item, int) or isinstance(item, slice)):
            raise ValueError("Can only pass int or slices.")

        s = item if isinstance(item, slice) else slice(item, item + 1)

        if s.step is not None and s.step != 1:
            raise ValueError("Step different than 1 is not supported")
        start = s.start + self._left_offset
        stop = s.stop + (s.stop % 2)
        return BitPackedSequence(
            self._bitpacked_seq[:,start / 2, stop /2],
            s.stop - s.start,
            start % 2
        )


    def to_tensor(self):
        #Because bit-packing is done 8 by 8:
        # * odd-length sequences are padded with an extra element
        # * Actual sequence sometime starts at element 1 (e.g. after slicing)
        unpacked_length = self._bitpacked_seq.shape[1] * 2
        unpacked = np.unpackbits(self._bitpacked_seq).reshape(unpacked_length,4)
        right_offset = -1 if self._left_offset + self._length % 2 == 1 else 0
        return unpacked[self._left_offset:right_offset]




class FastaTensor:
    _BASES = ['A', 'C', 'G', 'T']
    _contigs = {}

    def loadFasta(self, input):
        logger.info("Reading fasata file ".format(input))
        fasta = SeqIO.to_dict(SeqIO.parse(input, "fasta"))
        fasta = {a:b for a,b in fasta.iteritems() if a in ["1","20"]}
        encoder = LabelBinarizer(sparse_output=True).fit(self._BASES)
        for chrom, sequence in fasta.iteritems():
            logger.info("Encoding chromosome {}".format(chrom))
            one_hot = encoder.transform(list(str(sequence.seq)))
            self._contigs[chrom] = one_hot.astype(np.int8).todense()



