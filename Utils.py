import numpy as np


class Dot:

    def __init__(self, frame_size: int = 1000, coor: np.ndarray = np.array([0, 0])) -> None:
        """
        Инициализация класса, моделирующего случайное движение точки в пределах указанного фрейма,
        длина шага - один пиксель.
        :param frame_size: размер стороны квадратного фрейма
        :param coor: стартовые координаты точки
        """
        self.frame_size = frame_size
        self.coor = coor
        self.rnd = np.random.default_rng()

    def move(self) -> np.ndarray:
        """
        Формирование случайного вектора перемещения и расчет новых координат точки.
        :return: координаты точки после шага
        """
        while True:
            move_vector = self.rnd.integers(low=-1, high=2, size=2)
            curr = self.coor + move_vector

            flag = True
            for xi in curr:
                if xi < 0 or xi >= self.frame_size:
                    flag = False
            if flag:
                self.coor = curr
                return curr


class LinearMotionFilter:

    def __init__(self, n: int = 6) -> None:
        """
        Инициализация класса для предсказания координат по предыдущим значениям.
        :param n: количество предыдущих значений, используемых для расчета
        """
        self.n = n
        self.history = list()

    def update(self, coor: np.ndarray) -> None:
        """
        Обновление стека предыдущих положений точки.
        :param coor: новая пара коориднат точки
        """
        if len(self.history) > self.n:
            self.history.pop(0)
        self.history.append(coor)

    def predict(self) -> np.ndarray:
        """
        Расчет новой координаты:
        - из списка прошлых координат (x, y) длиной n формируется список перемещений (x_i - x_i-1, y_i - y_i-1)
          длины n - 1, оторый равен списку скоростей на этих перемещениях так как время берем за 1
        - рассчитываются средние скорости по каждой координате
        - расчет форомулы (x_predicted, y_predicted) = (x_last, y_last) + (speed_average_x, speed_average_y)*1
        :return: новая координата точки
        """
        if len(self.history) < self.n:
            return None

        speeds = list()
        for ind in range(1, len(self.history)):
            speeds.append(self.history[ind] - self.history[ind - 1])
        average_speed = np.mean(np.array(speeds), axis=0)
        return self.history[-1] + np.around(average_speed)
