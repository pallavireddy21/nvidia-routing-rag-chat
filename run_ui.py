"""Start the Chainlit UI: python run_ui.py  (works on Python 3.14)."""
import sys
import nest_asyncio
nest_asyncio.apply = lambda *a, **k: None  # nest_asyncio breaks asyncio on Python 3.14
from chainlit.cli import cli
if __name__ == "__main__":
    sys.argv = ["chainlit", "run", "app.py", *sys.argv[1:]]
    cli()
