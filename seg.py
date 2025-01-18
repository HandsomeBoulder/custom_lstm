from itertools import product
letters = "GBRY"
nums = "1234"
levels = [ch + num for num, ch in product(nums, letters)]
level_codes = [2 ** i for i in range(len(levels))]
code_to_level = {i: j for i, j in zip(level_codes, levels)}
level_to_code = {j: i for i, j in zip(level_codes, levels)}

def read_seg(filename: str, encoding: str = "utf-8-sig") -> tuple[dict, list[dict]]:
    with open(filename, encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines()]

    # найдём границы секций в списке строк:
    header_start = lines.index("[PARAMETERS]") + 1
    data_start = lines.index("[LABELS]") + 1

    # прочитаем параметры
    params = {}
    for line in lines[header_start:data_start - 1]:
        key, value = line.split("=")
        params[key] = int(value)

    # прочитаем метки
    labels = []
    for line in lines[data_start:]:
        # если в строке нет запятых, значит, это не метка и метки закончились
        if line.count(",") < 2:
            break
        pos, level, name = line.split(",", maxsplit=2)
        label = {
            "position": int(pos) // params["BYTE_PER_SAMPLE"] // params["N_CHANNEL"],
            "level": code_to_level[int(level)],
            "name": name
        }
        labels.append(label)
    return params, labels

def match_words_to_sounds(filename_upper, filename_lower, res_filename="res.txt"):
    _, labels_upper = read_seg(filename_upper, encoding="cp1251")  # Чтение Y1
    _, labels_lower = read_seg(filename_lower)  # Чтение В1
    res = []  # Инициализация итогового списка
    ctr = 0
    for start, end in zip(labels_upper, labels_upper[1:]):
        if not start["name"]:
            continue  # паузы нас не интересуют
        labels = []
        for label in labels_lower[ctr:]:
            if start["position"] <= label["position"] < end["position"]:
                ctr += 1
                labels.append(label)
            elif end["position"] <= label["position"]:  # оптимизация
                break
        label_names = []
        for i in labels:
            phoneme = i["name"]
            if phoneme != '~':
                if phoneme[-1].isdigit() or phoneme[-1] in ("_", "'"):
                    phoneme = phoneme[:-1]
                label_names.append(phoneme)
        
        res.append(label_names)

    return res