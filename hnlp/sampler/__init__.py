from hnlp.config import ARCH
if ARCH == "tf":
    from hnlp.sampler.sampler_tf import gen_input, gen_hidden
else:
    raise NotImplemented
