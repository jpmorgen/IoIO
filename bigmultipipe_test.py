import os
from tempfile import TemporaryDirectory, TemporaryFile
import numpy as np

from bigmultipipe import BigMultiPipe, prune_pout

class DemoMultiPipe(BigMultiPipe):

    def file_reader(self, in_name, **kwargs):
        data = np.load(in_name)
        return data

    def file_writer(self, data, outname, **kwargs):
        np.save(outname, data)
        return outname
    

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
        # --> This is equivalent to the keyword parameter
        # need_to_boost_by= boost_amount
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

with TemporaryDirectory() as tmpdirname:
    print('created temporary directory', tmpdirname)
    in_names = []
    print(in_names)
    for i in range(10):
        outname = f'big_array_{i}.npy'
        outname = os.path.join(tmpdirname, outname)
        a = i + np.zeros((1000,2000))
        np.save(outname, a)
        in_names.append(outname)
    dmp = DemoMultiPipe(pre_process_list=[reject, boost_later],
                        post_process_list=[later_booster, average],
                        outdir=tmpdirname)
    pout = dmp.pipeline(in_names, reject_value=2,
                        boost_target=3,
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
print(pruned_meta)
print(pruned_in_names)

