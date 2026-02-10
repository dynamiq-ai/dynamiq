"""Helper script that runs inside the sandbox to handle file operations.

This script is uploaded to the sandbox at runtime and invoked via shell commands.
It communicates results back as JSON on stdout.

Usage:
    python3 _file_helper.py read <path>
    python3 _file_helper.py write <path>   (content is read from stdin)
"""

import json
import os
import sys


def read_file(path):
    """Read file content and return as JSON."""
    try:
        with open(path) as f:
            content = f.read()
        result = {"ok": True, "content": content, "size": len(content)}
    except UnicodeDecodeError:
        size = os.path.getsize(path)
        result = {"ok": True, "content": f"(binary file, {size} bytes)", "size": size}
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    print(json.dumps(result))


def write_file(path, content):
    """Write content to a file and return result as JSON."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        result = {"ok": True, "size": len(content)}
    except Exception as e:
        result = {"ok": False, "error": str(e)}
    print(json.dumps(result))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "read":
        read_file(sys.argv[2])
    elif action == "write":
        # Content is passed via stdin to avoid shell escaping issues
        content = sys.stdin.read()
        write_file(sys.argv[2], content)
    else:
        print(json.dumps({"ok": False, "error": f"Unknown action: {action}"}))
