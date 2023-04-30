import cython
import numpy as np
cimport numpy as np


cdef class BoardConsts:
    APPLE = 2
    WALL = -1
    EMPTY = 0
    SNAKE_BODY = -2
    SNAKE_HEAD = 1

cpdef int chebyshev_distance(int[:] a, int[:] b):
    cdef int dx = abs(a[0] - b[0])
    cdef int dy = abs(a[1] - b[1])
    return max(dx, dy)

cdef class VisionLine:
    cdef double wall_distance
    cdef double apple_distance
    cdef double segment_distance

    def __init__(self, double wall_output, double apple_output, double segment_output):
        self.wall_distance = wall_output
        self.apple_distance = apple_output
        self.segment_distance = segment_output

    @property
    def wall_dist(self):
        return self.wall_distance

    @property
    def apple_dist(self):
        return self.apple_distance

    @property
    def segment_dist(self):
        return self.segment_distance

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_vision_lines_snake_head(np.ndarray[np.int32_t, ndim=2] board, np.ndarray[np.int32_t, ndim=1] snake_head,int vision_direction_count, str apple_return_type, str segment_return_type):
    cdef int[:,:] directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
    if vision_direction_count == 8:
        directions = np.append(directions, np.array([[-1, 1], [-1, -1], [1, -1], [1, 1]], dtype=np.int32), axis=0)

    cdef list vision_lines = []

    cdef int x_offset, y_offset
    cdef int[:] apple_coord, segment_coord, current_block, wall_coord
    cdef float wall_output, apple_output, segment_output
    cdef bint apple_found = False
    cdef bint segment_found = False

    for direction_tuple in directions:
        current_block = np.empty(2, dtype=np.int32)
        apple_coord = np.empty(2, dtype=np.int32)
        wall_coord = np.empty(2, dtype=np.int32)
        segment_coord = np.empty(2, dtype=np.int32)

        apple_found = False
        segment_found = False

        x_offset = direction_tuple[0]
        y_offset = direction_tuple[1]

        # search starts at one block in the given direction otherwise head is also check in the loop
        current_block[0] = snake_head[0] + x_offset
        current_block[1] = snake_head[1] + y_offset

        # loop the blocks in the given direction and store position and coordinates
        while board[current_block[0], current_block[1]] != -1:
            if board[current_block[0], current_block[1]] == 2 and apple_found == False:
                apple_coord[0] = current_block[0]
                apple_coord[1] = current_block[1]
                apple_found = True
            elif board[current_block[0], current_block[1]] == -2 and segment_found == False:
                segment_coord[0] = current_block[0]
                segment_coord[1] = current_block[1]
                segment_found = True

            current_block[0] = current_block[0] + x_offset
            current_block[1] = current_block[1] + y_offset

        wall_coord[0] = current_block[0]
        wall_coord[1] = current_block[1]
        wall_output = 1.0 / chebyshev_distance(snake_head, wall_coord)

        if apple_return_type == "boolean":
            apple_output = 1.0 if apple_found else 0.0
        else:
            apple_output = 1.0 / chebyshev_distance(snake_head, apple_coord) if apple_found else 0.0

        if segment_return_type == "boolean":
            segment_output = 1.0 if segment_found else 0.0
        else:
            segment_output = 1.0 / chebyshev_distance(snake_head, segment_coord) if segment_found else 0.0

        vision_lines.append(VisionLine(wall_output, apple_output, segment_output))

    return vision_lines
