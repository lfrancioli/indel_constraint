from utils import *
from hail import *

hc = HailContext()

vds = hc.read(final_genome_split_vds)
vds = vds.filter_intervals(Interval.parse("20:1-100000000"))
vds = vds.filter_variants_expr('v.altAllele.isIndel')
vds = vds.annotate_variants_expr('va.info = select(va.info, AC, AN)')
vds.export_vcf("gs://gnomad-lfran/tmp/gnomad_genomes_chrom20_indels.vcf")


