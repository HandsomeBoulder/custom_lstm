from seg import match_words_to_sounds  # функция обработки данных
from path_extractor import collect_paths  # функция формирования путей 
import numpy as np
from trask import *  # Кастомные функции для работы с построения модели и ее обучения
import matplotlib.pyplot as plt  # Библиотека для составления графиков

# Формирование путей
dirname = "cta_seg"  # Имя директории с корпусом файлов внутри
pathsB1, pathsY1 = collect_paths(dirname)  # Экстрация путей

# Формирование глобального списка фонем последовательно извлеченных из каждого файла
tokens = []
for B1, Y1 in zip(pathsB1, pathsY1):
    try:
        words = match_words_to_sounds(Y1, B1)
    except:
        continue
    for word in words:  # Работа с уровнем слов
        if len(word) != 0:
            for phoneme in word:  # Извлечение фонем из каждого слова
                if phoneme != '':  # Пропуски не интересуют
                    tokens.append(phoneme)  # Последовательная запись каждой фонемы в список

print("Общее кол-во аллофонов по всем файлам: ", len(tokens))

# Формирование множества уникальных фонем корпуса
vocab = set()  # Инициализация пустого множества
for phoneme in tokens:
    vocab.add(phoneme)  # Запись каждой итерации фонемы в множество, где она будет существовать только в единственной форме
vocab = list(vocab)  # Конвертация множества в список, чтобы с ним было удобнее работать в дальнейшем

# Выдача каждой фонеме уникального индекса
phon2index = {}  # Инициализация пустого словаря
for i,word in enumerate(vocab):  # Каждая фонема исчисляется
    phon2index[word]=i  # Ключ -- фонема, значение -- порядковый номер. Это удобно, потому что итерация по списку vocab будет выдавать тот же номер, что и поиск по словарю

# Формирование списков со смещением, альтернатива полноценным словам, позволяет игнорировать пэддинг
step = 1  # Шаг смещения
frame = 10  # Длина окна
tokens = [tokens[i:i+frame] for i in range(0, len(tokens) - frame - 1, step)]  # Кажду итерацию цикла окно смещается на step и отрезает участок списка длинной frame

# Замена фонем в глобальном списке окон на их числовые эквиваленты
indices = list()  # Инициализация пустого списка последовательностей
for token in tokens:  # Для каждой последовательности
    idx = list()  # Инициализация пустой последовательности индексов
    for phoneme in token:  # Фонему каждой последовательности
        idx.append(phon2index[phoneme])  # Конвертировать в индек и добавить в последовательность индексов
    indices.append(idx)  # Последовательность индексов добавить в список последовательностей

print("Общее кол-во последовательностей: ", len(indices))

# Составление выборок
data = np.array(indices)  # Превращения списка последовательностей в нампай матрицу
test_volume = 1000  # Объем тестовой выборки
train_data = data[:len(data)-test_volume]  # тренировочная выборка
test_data = data[len(data)-test_volume:]  # тестовая выборка

print("Параметры тренировочной выборки: ", train_data.shape)
print("Параметры тестовой выборки: ", test_data.shape)

# Построение модели
embed = Embedding(vocab_size=len(vocab),dim=128)  # Эмбеддинг слой для встраивания фонем
model = LSTMCell(n_inputs=128, n_hidden=256, n_output=len(vocab))  # LSTM ячейка для обработки последовательностей фонем
criterion = CrossEntropyLoss()  # Функция потерь
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(), alpha=0.005)  # Оптимизатор
train_accuracies = []  # Инициализация списка точностей по итерациям
total_iter = 1000  # Кол-во итераций

# Обучение модели
for iter in range(total_iter):
    batch_size = 2  # Кол-во последовательностей, который обрабатываются одновременно
    total_loss = 0  # Инициализация переменной, которая хранит потери
    hidden = model.init_hidden(batch_size=batch_size)  # Инициализация скрытых состояний LSTM для каждой партии

    for t in range(frame - 1):  # Проходит по последовательности данных, каждое t -- это номер фонемы в последовательности
        input = Tensor(train_data[0:batch_size,t], autograd=True)  # Забирает t фонему из текущей последовательности
        input = embed.forward(input=input)  # Преобразует числовой индекс фонемы в векторное представление
        output, hidden = model.forward(input=input, hidden=hidden) # Выполняет шаг LSTM, обновляет скрытые состояния и вычисляет выход модели

    target = Tensor(train_data[0:batch_size,t+1], autograd=True)  # Реальные метки следующего элемента последовательности
    loss = criterion.forward(output, target)  # Вычисление потерь между предсказанным выходом и истинным значением
    loss.backward()  # Обратное распространение ошибки для обновления градиентов
    optim.step()  # Приминение оптимизатора для обновления весов модели
    total_loss += loss.data  # Суммирует потери для отслеживания эффективности
    if(iter % 200 == 0):  # Каждые 200 итераций выводит среднее значение потерь и точности
        train_accuracy = calculate_accuracy(train_data, model, embed, vocab)  # Вычисление точности
        train_accuracies.append(train_accuracy)  # Добавление точности в список точностей для построения графика
        print("Loss:",total_loss / (len(train_data)/batch_size), "Train accuracy: ", train_accuracy)  


# Оценка точности работы сети
correct_predictions = 0  # Инициализация переменных и списков
total_predictions = 0
predicted = []
real = []

for seq in test_data:  # Итерация по тестовым последовательностям
    context = seq[:-1]  # Контекст -- это все, кроме последнего элемента
    true_next_sound = seq[-1]  # Реальный последний элемент последовательности
    
    hidden = model.init_hidden(batch_size=1)  # Инициализирует скрытые состояния LSTM для текущей последовательности
    for t in range(len(context)):  # Каждый элемент -- фонема, которая подается модели последовательно
        input = Tensor([context[t]], autograd=True)  # Преобразует текущую фонему в тензор
        lstm_input = embed.forward(input)  # Преобразует числовой индекс фонемы в векторное представление
        output, hidden = model.forward(lstm_input, hidden)  # Передает векторное представление и скрытые состояни в ЛСТМ

    predicted_sound_index = np.argmax(output.data)  # Определяет инекс фонемы по наибольшему значению в выходных данных
    if predicted_sound_index == true_next_sound:  # Сравнение предсказ. значения с истинным
        correct_predictions += 1  # Если они совпадают, то увеличивает счетчик правильных предсказаний
        predicted.append(vocab[predicted_sound_index])  # Добавление ошибочных предсказаний и реальных значений в списки для отслеживания
        real.append(vocab[true_next_sound])
    else:
        predicted.append(vocab[predicted_sound_index])  # Добавление ошибочных предсказаний и реальных значений в списки для отслеживания
        real.append(vocab[true_next_sound])
    total_predictions += 1  # Счетчик общего кол-ва предсказаний
accuracy = correct_predictions / total_predictions  # Точность сети на тестовой выборке
for r, p in zip(real[:10], predicted[:10]):  # Демонстрация первых 10 предсказаний и ответов
    print(f"Predicted: {p}; Real: {r}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Формирование графика повышения точности при обучении
iters = list(range(1, total_iter//200+1))  # Количетсво контрольных точек для подсчета точности во время обучения
plt.plot(iters, train_accuracies, label='Train Accuracy')
plt.xlabel('Every 200 iterations')
plt.ylabel('Accuracy')
plt.title('Accuracy over iterations')
plt.legend()
plt.show()