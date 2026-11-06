"""Process-scoped paths. Set the environment before importing pipeline modules."""
import os


def project_output(path='output'):
    root = os.environ.get('VL_OUTPUT_DIR')
    if not root:
        return path
    return root.replace('\\', '/') + path[len('output'):]


def subtitle_filter_path(path):
    """Escape filename for both filtergraph and subtitles option parsers."""
    # Escape the option value first, then the surrounding filtergraph value.
    value = str(path).replace('\\', '/')
    value = ''.join('\\'+c if c in "\\':" else c for c in value)
    return ''.join('\\'+c if c in "\\'[],; " else c for c in value)
