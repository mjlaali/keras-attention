import tensorflow as tf
import tensorflow.contrib.eager as tfe


def run_test_in_eager_mode(func):
    """
    We cannot enable eager mode in the tests because it will affect other tests too. This happens
    because, unfortunately, pytest does not separate running tests properly. Check this issue for
    more information:
    https://stackoverflow.com/questions/50143896/both-eager-and-graph-execution-in-tensorflow-tests
    A work around is to run these tests without enabling eager mode, but use tfe.py_fun to run
    the tests.
    Parameters
    ----------
    func: the tests which need to be run in eager mode.

    Returns
    -------
    a decorated test which does not need eager mode.

    """

    def decorator():
        with tf.Session() as sess:
            sess.run(tfe.py_func(func, inp=[], Tout=[]))

    return decorator