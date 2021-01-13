"""bigmultipipe example code showing evolution from traditional for loop

"""

import os
from tempfile import TemporaryDirectory, TemporaryFile
import numpy as np

from bigmultipipe import BigMultiPipe, prune_pout


# Traditional for loop case

# Write some large files
with TemporaryDirectory() as tmpdirname:
    print(f'created temporary directory {tmpdirname} for sample raw data files')
    in_names = []
    for i in range(10):
        outname = f'big_array_{i}.npy'
        outname = os.path.join(tmpdirname, outname)
        a = i + np.zeros((1000,2000))
        np.save(outname, a)
        in_names.append(outname)

    # Process with traditional for loop
    reject_value = 2
    boost_target=3
    boost_amount=5
    outnames = []
    meta = []
    for f in in_names:
        # File read step
        data = np.load(f)
        # Pre-processing steps
        flag_to_boost_later = False
        if data[0,0] == reject_value: 
            continue
        if data[0,0] == boost_target:
            flag_to_boost_later = True
        # Processing step
        data = data * 10
        # Post-processing steps
        if flag_to_boost_later:
            data = data + boost_amount
        meta.append({'average': np.average(data)})
        outname = f + '_bmp'
        np.save(outname, data)
        outnames.append(outname)
    print(outnames, meta)


# BigMultiPipe object for parallel processing above code
class DemoMultiPipe(BigMultiPipe):

    def file_read(self, in_name, **kwargs):
        data = np.load(in_name)
        return data

    def file_write(self, data, outname, **kwargs):
        np.save(outname, data)
        return outname

    def data_process(self, data, **kwargs):
        return data * 10
    
# Define hook-like routines for our BigMultiPipe pre_process_list and
# post_process_list property.  These routines could be incorporated in
# the data_process method, but doing it this way enables the pipeline
# construction and control features noted below
def reject(data, reject_value=None, **kwargs):
    """Example pre-processing function to reject data"""
    if reject_value is None:
        return (data, {})
    if data[0,0] == reject_value:
        # --> Return data=None to reject data
        return (None, {})
    return (data, {})

def boost_later(data, boost_target=None, boost_amount=None, **kwargs):
    """Example pre-processing function that shows how to alter kwargs"""
    if boost_target is None or boost_amount is None:
        return (data, {})
    if data[0,0] == boost_target:
        # --> This is equivalent to setting a keyword parameter
        # need_to_boost_by=boost_amount
        return (data, {'need_to_boost_by': boost_amount})
    return (data, {})

def later_booster(data, meta, need_to_boost_by=None, **kwargs):
    """Example post-processing function.  Interprets keyword set by boost_later"""
    if need_to_boost_by is None:
        return (data, {})
    data = data + need_to_boost_by
    return (data, {})

def average(data, meta, **kwargs):
    """Example metadata generator"""
    av = np.average(data)
    return (data, {'average': av})

# Example code using bigmultipipe instead of for loop
with TemporaryDirectory() as tmpdirname:
    print(f'created temporary directory {tmpdirname} for sample raw and processed data files')
    in_names = []
    for i in range(10):
        outname = f'big_array_{i}.npy'
        outname = os.path.join(tmpdirname, outname)
        a = i + np.zeros((1000,2000))
        np.save(outname, a)
        in_names.append(outname)

    # Create a pipeline using the pre- and post-processing
    # components defined above.  This enables pipeline is to be
    # assembled at instantiation and controlled at either
    # instantiation or runtime 
    dmp = DemoMultiPipe(pre_process_list=[reject, boost_later],
                        post_process_list=[later_booster, average],
                        boost_target=3, outdir=tmpdirname)
    pout = dmp.pipeline(in_names, reject_value=2,
                        boost_amount=5)
outnames, meta = zip(*pout)
print(outnames)
print(meta)

print('Pruning outname ``None`` and removing directory')
pruned_pout, pruned_in_names = prune_pout(pout, in_names)
pruned_outnames, pruned_meta = zip(*pruned_pout)
pruned_outnames = [os.path.basename(f) for f in pruned_outnames]
pruned_in_names = [os.path.basename(f) for f in pruned_in_names]
print(pruned_outnames)
print(pruned_in_names)

print(pruned_meta)

