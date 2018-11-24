import logging as log
import pandas as pd

def optimize_dataset(path, reuse = 0.15, output = None):
    log.info("optimizing dataset %s (reuse ratio is %.1f%%) ...", path, reuse * 100.0)

    data  = pd.read_csv(path, sep = ',', header = None)
    n_tot = len(data)

    log.info("loaded %d total samples", n_tot)

    unique  = data.drop_duplicates()
    n_uniq  = len(unique)
    n_reuse = int( n_uniq * reuse )
    reuse   = data.sample(n=n_reuse).reset_index(drop = True)

    log.info("found %d unique samples, reusing %d samples from the main dataset", n_uniq, n_reuse)

    out          = pd.concat([reuse, unique]).sample(frac=1).reset_index(drop=True)
    outpath      = output if output is not None else path
    n_out        = len(out)
    optimization = 100.0 - (n_out * 100.0) / float(n_tot)

    log.info("optimized dataset has %d records, optimization is %.2f%%", n_out, optimization)

    log.info("saving %s ...", outpath)
    out.to_csv( outpath, sep = ',', header = None, index = None)
