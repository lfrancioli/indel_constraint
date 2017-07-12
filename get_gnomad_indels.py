from utils import *
from hail import *

hc = HailContext()

vds = hc.read(final_genome_split_vds_path)
vds = vds.filter_variants_expr('v.altAllele.isIndel')
vds = vds.annotate_variants_expr('va.info = select(va.info, AC, AN)')
for i in range(2,23):
    chrom = vds.filter_intervals(Interval.parse("{}:1-300000000".format(i)))
    chrom.export_vcf("gs://gnomad-multi_regional/indel_constraint/vcf/gnomad_genomes_chrom{}_indels.vcf".format(i))


