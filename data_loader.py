from pathlib import Path


def _find_text_files(folder_path: str, file_name: str = 'text.txt') -> list[Path]:
    folder = Path(folder_path)
    res = list(folder.rglob(file_name))
    res.sort()
    return res


def load_data(folder_path: str, file_name: str) -> list[tuple[Path, str]]:
    files = _find_text_files(folder_path,  file_name)
    if not files:
        print(f"No files found in {folder_path} with name {file_name}")
        return []

    data: list[tuple[Path, str]] = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            data.append((file, f.read()))

    return data


def list_subfolders(folder_path: str) -> list[Path]:
    folder = Path(folder_path)
    return [f for f in folder.iterdir() if f.is_dir()]
