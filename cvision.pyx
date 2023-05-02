#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False

cpdef get_all_random_blocks(int[:, :] board,int rows,int cols):
    cdef list empty = []

    for i in range(1,rows):
        for j in range(1,cols):
            if board[i][j] == 0:
                empty.append([i,j])

    return empty

cdef struct Pair:
     int x
     int y

cdef class VisionLine:
    cdef public Pair wall_coord
    cdef public double wall_distance
    cdef public Pair apple_coord
    cdef public double apple_distance
    cdef public Pair segment_coord
    cdef public double segment_distance

    def __init__(self,Pair wall_coord, double wall_output,Pair apple_coord, double apple_output,Pair segment_coord, double segment_output):
        self.wall_coord = wall_coord
        self.wall_distance = wall_output
        self.apple_coord = apple_coord
        self.apple_distance = apple_output
        self.segment_coord = segment_coord
        self.segment_distance = segment_output



cpdef get_vision_lines_snake_head(int[:, :] board, int[:] snake_head,int vision_direction_count, str apple_return_type, str segment_return_type):

    cdef int directions[8][2]
    directions[0][0] = -1
    directions[0][1] = 0
    directions[1][0] = 1
    directions[1][1] = 0
    directions[2][0] = 0
    directions[2][1] = -1
    directions[3][0] = 0
    directions[3][1] = 1
    directions[4][0] = -1
    directions[4][1] = 1
    directions[5][0] = -1
    directions[5][1] = -1
    directions[6][0] = 1
    directions[6][1] = -1
    directions[7][0] = 1
    directions[7][1] = 1

    cdef list vision_lines = []
    cdef int x_offset, y_offset
    cdef Pair apple_coord, segment_coord, current_block, wall_coord

    cdef float wall_output, apple_output, segment_output, output_distance
    cdef int board_element
    cdef bint apple_found = False
    cdef bint segment_found = False
    cdef bint wall_found = False
    cdef int dx
    cdef int dy

    for i in range(vision_direction_count):
        apple_coord.x = 0
        apple_coord.y = 0
        segment_coord.x = 0
        segment_coord.y = 0
        current_block.x = 0
        current_block.y = 0
        wall_coord.x = 0
        wall_coord.y = 0

        apple_found = False
        segment_found = False
        wall_found = False

        x_offset = directions[i][0]
        y_offset = directions[i][1]

        # search starts at one block in the given direction otherwise head is also check in the loop
        current_block.x = snake_head[0] + x_offset
        current_block.y = snake_head[1] + y_offset
        board_element = board[current_block.x, current_block.y]

        if board_element == -2:
            segment_coord = current_block
            segment_found = True

        if board_element == -1:
            wall_coord = current_block
            wall_found = True

        # loop the blocks in the given direction and store position and coordinates
        while board_element != -1:
            if board_element == 2 and apple_found == False:
                apple_coord = current_block
                apple_found = True
            current_block.x += x_offset
            current_block.y += y_offset
            board_element = board[current_block.x, current_block.y]

        if wall_found:
            dx = abs(snake_head[0] - wall_coord.x)
            dy = abs(snake_head[1] - wall_coord.y)
            output_distance =  max(dx, dy)
            wall_output = 1.0 / output_distance
        else:
            wall_output = 0.0

        if apple_return_type == "boolean":
            apple_output = 1.0 if apple_found else 0.0
        else:
            if apple_found:
                dx = abs(snake_head[0] - apple_coord.x)
                dy = abs(snake_head[1] - apple_coord.y)
                output_distance =  max(dx, dy)
                apple_output = 1.0 / output_distance

            else:
                apple_output = 0.0

        if segment_return_type == "boolean":
            segment_output = 1.0 if segment_found else 0.0
        else:
            if segment_found:
                dx = abs(snake_head[0] - segment_coord.x)
                dy = abs(snake_head[1] - segment_coord.y)
                output_distance =  max(dx, dy)

                segment_output = 1.0/output_distance
            else:
                segment_output = 0.0

        vision_lines.append(VisionLine(wall_coord,wall_output, apple_coord,apple_output,segment_coord, segment_output))

    return vision_lines

cpdef update_board_from_snake(int[:, :] board,int[:,:]body):
    cdef int x, y
    cdef int width = board.shape[0]
    cdef int height = board.shape[1]
    for x in range(width):
        for y in range(height):
            board_val = board[x,y]
            if board_val == 1:
                board[x,y] = 0
            if board_val == -2:
                board[x, y] = 0

    board[body[0][0],body[0][1]] = 1
    for i in range(1,len(body)):
        board[body[i][0],body[i][1]] = -2

    return board
