def assert_broadcastable(shape1, shape2):
    if len(shape1) > len(shape2):
        assert_broadcastable(shape2, shape1)
        return
    for a,b in zip(shape1[::-1], shape2[::-1][:len(shape1)]):
        assert (a == 1 or b == 1 or a == b), "The shape %s and %s are not broadcastable" % (shape1, shape2)

def assert_runtime(cond, msg=""):
    if not cond:
        raise RuntimeError(msg)

def assert_type(cond, msg=""):
    if not cond:
        raise TypeError(msg)