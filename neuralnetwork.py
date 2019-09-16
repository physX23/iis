import os
import random
import matplotlib.pyplot
import scipy.special
import numpy
from matplotlib.widgets import Button

matplotlib.use('TkAgg')


# определяем класс нейронной сети
class NeuralNetwork:

    # инициализируем нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задаем количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes  # входной слой
        self.hnodes = hiddennodes  # скрытый слой
        self.onodes = outputnodes  # выходной слой

        # матрицы весовых коэффициентов связей, wih и who
        # Весовые коэффициенты связей между узлом i и узлом j
        # w11 w21
        # w12 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # коэффициент обучения
        self.lr = learningrate

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # расчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # расчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # расчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # вычисление ошибки
        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs
        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # обновить весовые коэффициенты связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразование входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # расчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # расчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # расчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # расчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def load_weight(self):
        if os.path.exists('./wih.npy') and os.path.exists('./who.npy'):
            self.wih = numpy.load('wih.npy')
            self.who = numpy.load('who.npy')
            return True
        else:
            return False

    def save_weight(self):
        numpy.save('wih.npy', self.wih)
        numpy.save('who.npy', self.who)


# количество входных, скрытых и выходных узлов
input_nodes = 784  # так как картинки 28 на 28 пискелей, то входных нейронов = 28 * 28
hidden_nodes = 200
output_nodes = 10  # выходных значений 10, по количеству классов одежды
clothes_labels = ["Футболка / Топ", "Шорты", "Свитер",
                  "Платье", "Плащ", "Сандали", "Рубашка",
                  "Кросовки", "Сумка", "Ботинок"]
# коэффициент обучения
learning_rate = 0.01

# создаем экземпляр нейронной сети
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# загружаем в список тренировочный набор данных CSV-файла набора MNIST
training_data_file = open("data/clother/mnist_train_60k.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# тренировка нейронной сети
#
# переменная epochs (эпохи) указывает, сколько раз тренировочный
# набор данных используется для тренировки сети

epochs = 7
if not n.load_weight():
    for e in range(epochs):
        print(f'Эпоха №{e+1}')
        #  осуществляем перебор всех записей в тренеровочном наборе
        for record in training_data_list:
            # получаем список значений, используя симвой запятой (',') в качестве разделителя
            all_values = record.split(',')
            # масштабируем и смещаем входные значения, иными словами приводим входные значения к диапазону 0,01 - 1,00
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # создаем целевые выходные значения
            # (все равные 0,1, за исключением желаемого маркерного значения, равного 0,99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] - целевое маркерное значение для данной записи
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
    print(f'Сохраняем состояние нейронной сети')
    n.save_weight()

# загружаем в список тестовый набор данных CSV-файла набора MNIST
test_data_file = open("data/clother/mnist_test_10k.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# тестирование нейронной сети

# журнал оценок работы сети, первоначально пустой
scorecard = []
print(f'Запуск тестирования нейронной сети')
# перебрать все записи в тестовом наборе данных
for record in test_data_list:
    # получаем список значений, используя симвой запятой (',') в качестве разделителя
    all_values = record.split(',')
    # правильный ответ - первое значение
    correct_label = int(all_values[0])
    # масштабируем и смещаем входные значения, иными словами приводим входные значения к диапазону 0,01 - 1,00
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # опрашиваем нейронную сеть
    outputs = n.query(inputs)
    # индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
    # проверям является ли ответ верным
    if label == correct_label:
        # в случае верного ответа, добавляем в журнал оценок 1
        scorecard.append(1)
    else:
        # в случае неверного ответа, добавляем в журнал оценок 0
        scorecard.append(0)
        pass

    pass

# рассчитать показатель эффективности в виде доли правильных ответов
scorecard_array = numpy.asarray(scorecard)
print(f"Эффективность '{((scorecard_array.sum() / scorecard_array.size)*100)}'%")


def open_window():
    matplotlib.pyplot.close()
    select_index_picture = random.randrange(0, 10000)
    all_values = test_data_list[select_index_picture].split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    right_label = clothes_labels[int(all_values[0])]

    fig, ax = matplotlib.pyplot.subplots(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle(f"Правильный ответ: {right_label} ({select_index_picture})", fontsize=14, fontweight='bold')

    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.gcf().canvas.set_window_title(right_label)

    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = clothes_labels[int(numpy.argmax(outputs))]

    matplotlib.pyplot.title(f'Нейронная сеть считает\nчто на изображении находится "{label}"\n'
                            f'с вероятностью {"{0:.2f}".format(numpy.amax(outputs)*100)}%')
    print(f'Нейронная сеть считает что на изображении находится "{label}" '
          f'с вероятностью {"{0:.2f}".format(numpy.amax(outputs)*100)}%')

    axnext = matplotlib.pyplot.axes([0.55, 0.01, 0.1, 0.075])
    bnext = Button(axnext, 'Далее')
    bnext.on_clicked(lambda x: open_window())

    matplotlib.pyplot.show()


open_window()


