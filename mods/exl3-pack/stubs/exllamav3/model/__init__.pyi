

# Model is referenced by convert_model/allocation signatures (as Any there), and
# the real model/__init__.py re-exports it. Declared minimally here so the
# top-level `from exllamav3.model import Model` resolves without stubbing the
# full Model implementation.
class Model: ...
