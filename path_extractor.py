import os.path

def collect_paths(dir_name: str) -> list:
    pathsB1, pathsY1 = [], []
    # Пути к файлам
    for address, _, files in os.walk(dir_name):
        for name in files:
            if ".seg_B1" in name:
                pathsB1.append(os.path.join(address, name))
            if ".seg_Y1" in name:
                pathsY1.append(os.path.join(address, name))

    return pathsB1, pathsY1

# print(collect_paths("text"))