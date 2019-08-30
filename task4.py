import cv2
import argparse
import numpy as np


def track_method_1(input_file_path, board_dims):
    cap = cv2.VideoCapture(input_file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners_ret, corners = cv2.findChessboardCorners(gray, board_dims, None)

        if corners_ret:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(gray, (x, y), 3, (0, 0, 255), -1)

        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def track_method_2(input_file_path, board_dims):
    cap = cv2.VideoCapture(input_file_path)

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))

    ret_corners = False
    while not ret_corners:
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        ret_corners, p0 = cv2.findChessboardCorners(old_gray, board_dims, None)
        print(ret_corners)
        mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


def main(input_file_path, output_dir, board_dims, track_method):
    if track_method == 'method_1':
        track_method_1(input_file_path, board_dims)
    else:
        track_method_2(input_file_path, board_dims)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='task2')
    parser.add_argument('--input-file-path',
                        default='./data/task4/chessboard_pattern_tech_interview.mp4',
                        help='path of input video')

    parser.add_argument('--output-dir',
                        default='./data',
                        help='')

    parser.add_argument('--board-dims',
                        default=[6, 10],
                        nargs="+",
                        type=int,
                        help='')

    parser.add_argument('--track-method',
                        choices=['method_1', 'method_2'],
                        default='method_2',
                        help='track method in assignment')

    args = parser.parse_args()
    args.board_dims = tuple(args.board_dims)
    main(args.input_file_path, args.output_dir, args.board_dims, args.track_method)
