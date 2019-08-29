import cv2
import numpy as np

def find_points(images):
    pattern_size = (9, 6)
    obj_points = []
    img_points = []

    # Assumed object points relation
    a_object_point = np.zeros((PATTERN_SIZE[1] * PATTERN_SIZE[0], 3),
                              np.float32)
    a_object_point[:, :2] = np.mgrid[0:PATTERN_SIZE[0],
                                     0:PATTERN_SIZE[1]].T.reshape(-1, 2)

    # Termination criteria for sub pixel corners refinement
    stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
                     30, 0.001)

    print('Finding points ', end='')
    debug_images = []
    for (image, color_image) in images:
        found, corners = cv.findChessboardCorners(image, PATTERN_SIZE, None)
        if found:
            obj_points.append(a_object_point)
            cv.cornerSubPix(image, corners, (11, 11), (-1, -1), stop_criteria)
            img_points.append(corners)

            print('.', end='')
        else:
            print('-', end='')

        if DEBUG:
            cv.drawChessboardCorners(color_image, PATTERN_SIZE, corners, found)
            debug_images.append(color_image)

        sys.stdout.flush()

    if DEBUG:
        display_images(debug_images, DISPLAY_SCALE)

    print('\nWas able to find points in %s images' % len(img_points))
    return obj_points, img_points



#cap = cv2.VideoCapture('./data/task4/chessboard_pattern_tech_interview.mp4')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     corners_ret, corners = cv2.findChessboardCorners(frame, (6,8))
#
#     cv2.imshow('frame', gray)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()



# cap = cv2.VideoCapture('./data/task4/chessboard_pattern_tech_interview.mp4')
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     corners_ret, corners = cv2.findChessboardCorners(frame, (6,8))
#
#     if corners_ret:
#         print(corners.shape)
#         corner = corners[0][0]
#         x, y = int(corner[0]), int(corner[1])
#         z, h = x+10, y+10
#         print(x)
#         print(y)
#         print(z)
#         print(h)
#
#         cv2.rectangle(gray, (x,y), (z,h), (255,255,255), 2)
#
#         cv2.imshow('frame', gray)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#
# cap.release()
# cv2.destroyAllWindows()



cap = cv2.VideoCapture('./data/task4/chessboard_pattern_tech_interview.mp4')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners_ret, corners = cv2.findChessboardCorners(frame, (3, 3))

    if corners_ret:
        for i in corners:
            x, y = i.ravel()
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

