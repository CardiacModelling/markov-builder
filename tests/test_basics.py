#
# Tests the basics of markov_builder
#

 
def test_module_import():
    import markov_builder  # noqa


def test_version():
    # Test the version() method
    import markov_builder as mm

    version = mm.version()
    assert isinstance(version, tuple)
    assert len(version) == 3
    assert isinstance(version[0], int)
    assert isinstance(version[1], int)
    assert isinstance(version[2], int)
    assert version[0] >= 0
    assert version[1] >= 0
    assert version[2] >= 0

    version = mm.version(formatted=True)
    assert isinstance(version, str)
    assert len(version) >= 1
    assert version.startswith('markov_builder ')
