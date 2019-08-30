import cv2
import argparse


def track_method_1(input_file_path):
    cap = cv2.VideoCapture(input_file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_ret, corners = cv2.findChessboardCorners(frame, (3, 3))

        if corners_ret:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main(input_file_path, output_dir):
    track_method_1(input_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='task2')
    parser.add_argument('--input-file-path',
                        default='./data/task2',
                        help='path of input video')

    parser.add_argument('--output-dir',
                        default='./data',
                        help='')

    args = parser.parse_args()

    main(args.input_file_path, args.output_dir)
