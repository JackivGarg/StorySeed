"""Read a bounded TinyStories prefix; a child deadline bounds network work."""
import subprocess
import sys

if __name__ == '__main__':
    if '--worker' in sys.argv:
        from storyseed.corpus import capture_prefix
        capture_prefix()
    else:
        subprocess.run([sys.executable, __file__, '--worker'], check=True, timeout=55)
