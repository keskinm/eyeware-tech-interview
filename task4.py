import cv2
import argparse


def track_method_1(input_file_path, board_dims):
    cap = cv2.VideoCapture(input_file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_ret, corners = cv2.findChessboardCorners(frame, board_dims)

        if corners_ret:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main(input_file_path, output_dir, board_dims):
    track_method_1(input_file_path, board_dims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='task2')
    parser.add_argument('--input-file-path',
                        default='./data/task4/chessboard_pattern_tech_interview.mp4',
                        help='path of input video')

    parser.add_argument('--output-dir',
                        default='./data',
                        help='')

    parser.add_argument('--board-dims',
                        default=(3, 3),
                        type=tuple,
                        help='')

    args = parser.parse_args()

    main(args.input_file_path, args.output_dir, args.board_dims)
