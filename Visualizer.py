import numpy as np
import cv2

from Utils import Dot, LinearMotionFilter


class Visualizer:

    def __init__(self, frame_size: int = 1000) -> None:
        self.frame_size = frame_size
        self.dot = Dot(frame_size=frame_size,
                       coor=np.array([frame_size // 2, frame_size // 2]))
        self.model = LinearMotionFilter()

    def video(self) -> None:
        last_predicted = None
        total = 0
        true = 0

        while True:
            frame = np.zeros(shape=(self.frame_size, self.frame_size))

            curr = self.dot.move()
            if last_predicted is not None:
                total += 1
                true += int(curr.tolist() == last_predicted.tolist())
            self.model.update(curr)
            last_predicted = self.model.predict()

            cv2.circle(frame, curr, 5, [255, 255, 255], -1)
            cv2.imshow('dot animation', frame)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                print(f'accuracy: {round(true / total, 2)}')
                break


if __name__ == '__main__':
    vis = Visualizer(frame_size=800)
    vis.video()
