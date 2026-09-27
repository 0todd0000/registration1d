'''
Make the package importable from a source checkout without installation
(src layout): tests import registration1d from ../src. PYTHONPATH is set
as well so that worker processes started by the parallel options (which
re-import the package under the 'spawn' start method used on macOS and
Windows) find it too. After `pip install -e .` neither is needed.
'''
import os, sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ['PYTHONPATH'] = SRC + os.pathsep + os.environ.get('PYTHONPATH', '')
# render Qt off-screen when the optional PyQtGraph backend is tested (harmless with a display)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
