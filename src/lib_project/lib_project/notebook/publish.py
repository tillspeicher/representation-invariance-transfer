import time
import uuid
from pathlib import Path

import nbformat
from IPython.core.display import Markdown as md
from IPython.core.display import display
from nbconvert import HTMLExporter

from .upload import get_upload_config, upload_files


def publish_notebook(
    nb_file: str,
    target_file: str,
    exclude_input: bool = True,
) -> None:
    with open(nb_file, "r", encoding="utf-8") as f:
        content = f.read()
        # Usually the file is saved just before calling this code.
        # Saving might take some time, during which the file is in an
        # invalid state and empty. We wait until the file is not empty
        # anymore before reading it.
        while len(content) == 0:
            time.sleep(2)
            content = f.read()
    notebook_node = nbformat.reads(content, as_version=4)

    exporter = HTMLExporter(
        template_name="classic", exclude_input=exclude_input
    )
    output, _ = exporter.from_notebook_node(notebook_node)

    intermediate_file = Path("/tmp/") / f"{uuid.uuid4()}.html"
    with open(intermediate_file, "w") as f:
        f.write(output)

    upload_files([str(intermediate_file)], [target_file])

    _, upload_config = get_upload_config()
    full_results_url = f"{upload_config.results_url_prefix}/{target_file}"

    display(md(f"Notebook published to: {full_results_url}"))
