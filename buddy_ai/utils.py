import pathlib
import logging

def read_file_or_directory(path_str: str) -> str:
    path = pathlib.Path(path_str)
    content_parts = []
    if path.is_file():
        try:
            file_content = path.read_text(encoding='utf-8', errors='ignore')
            content_parts.append(f"### Content from file: {path.name}\n```\n{file_content}\n```")
        except Exception as e:
            logging.error(f"Error reading file {path}: {e}")
            return f"Error reading file {path}: {e}"
    elif path.is_dir():
        content_parts.append(f"### Content from directory: {path.name}\n")
        found_files_in_dir = False
        allowed_extensions = [
            ".txt", ".py", ".md", ".sh", ".json", ".yaml", ".yml", ".h", ".c", ".cc", ".cpp", ".java",
            ".js", ".ts", ".html", ".css", ".rb", ".php", ".pl", ".tcl", ".go", ".rs", ".swift",
            ".kt", ".scala", ".r", ".ps1", ".psm1", ".bat", ".cmd", ".vb", ".vbs", ".sql", ".xml",
            ".ini", ".cfg", ".conf", ".toml", ".dockerfile", "Dockerfile", ".tf", ".tex", ".rtf",
            ".csv", ".log", ".err", ".out", ".xsd", ".wsdl", ".java", ".properties", ".gradle", ".gitignore"
        ]
        for item in path.rglob("*"):
            if item.is_file() and (item.suffix.lower() in allowed_extensions or item.name == "Dockerfile"):
                try:
                    file_content = item.read_text(encoding='utf-8', errors='ignore')
                    relative_file_path = item.relative_to(path) if path in item.parents else item.name
                    content_parts.append(f"#### File: {relative_file_path}\n```\n{file_content}\n```")
                    found_files_in_dir = True
                except Exception as e:
                    logging.error(f"Error reading file {item} in directory {path_str}: {e}")
        if not found_files_in_dir:
             content_parts.append("\nNo recognized text files found in directory.")
    else:
        logging.error(f"Path '{path_str}' is not a valid file or directory.")
        return f"Error: Path '{path_str}' is not a valid file or directory."
    return "\n\n".join(content_parts)
