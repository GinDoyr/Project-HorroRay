import arcade
import arcade.clock
import math
import numpy as np
from side_lib.gif_to_sprite import gifSprite
from pyglet.math import Vec2

# -----Options-----
NUM_RAYS = 90  # Must be between 1 and 90. might change later on
NUM_WALLS = 5  # The amount of randomly generated walls
MAX_DOTS = 10000  # heey finally one of my own :D max "lidar" generated dots
MIN_RAY_LEN = 300
BASE_DOT_SPEED = 5
START_PLAYER_SPEED = 3 # probably dont do less than 3
MUSIC_VOLUME = 0.5 # 0.00 to 1.00
SFX_VOLUME = 0.5
# ------------------

window = arcade.Window(1200, 600, 'Project: HorroRay')
window.center_window()
area = [(0, 0), (window.width, window.height)]
visited_areas = []

mx, my = 600, 300
ray_len = MIN_RAY_LEN
dot_speed = BASE_DOT_SPEED
lastClosestPoint = (0, 0)
curr_angle = 45
area_limit1 = 10
area_limit = 5
rays = []
walls = []
drawwalls = []
lidar_walls = []
walldots = []
precreaturedots = []
creaturedots = []
escapes = []
left = False
right = False
up = False
down = False
up1 = False
down1 = False
mup = False
mdown = False
mleft = False
mright = False
mrun = False
lidar_flag = False
ray_flag = False
bvh_flag = True
drawbvh = False
DEBUG_FLAG = False
game_over = False
win = False
ray_mode = True
can_escape = False
intersect_count = 0
stamina = 10000


class Ray:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle  # Store angle for sorting
        self.dir = (math.cos(angle), math.sin(angle))
        self.last_hit_wall = None  # Track last wall this ray hit
        self.end_x = ray_len * self.dir[0] + x
        self.end_y = ray_len * self.dir[1] + y

    def update(self, x, y):
        self.x = x
        self.y = y

    def checkCollision(self, wall):
        x1, y1 = wall.start_pos
        x2, y2 = wall.end_pos
        x3, y3 = self.x, self.y

        denominator = (x1 - x2) * (-self.dir[1]) - (y1 - y2) * (-self.dir[0])
        numerator = (x1 - x3) * (-self.dir[1]) - (y1 - y3) * (-self.dir[0])
        if denominator == 0:
            return None

        t = numerator / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 1 > t > 0 and u > 0:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            collidePos = [x, y]
            return collidePos

    def checkCreatureCollision(self, kreatura, closest):
        close = closest
        x3, y3 = self.x, self.y
        for i in range(len(kreatura.point_list)):
            if i == len(kreatura.point_list)-1:
                x1, y1 = kreatura.point_list[i]
                x2, y2 = kreatura.point_list[0]
            else:
                x1, y1 = kreatura.point_list[i]
                x2, y2 = kreatura.point_list[i+1]

            denominator = (x1 - x2) * (-self.dir[1]) - (y1 - y2) * (-self.dir[0])
            numerator = (x1 - x3) * (-self.dir[1]) - (y1 - y3) * (-self.dir[0])
            if denominator != 0:
                t = numerator / denominator
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

                if 1 > t > 0 and u > 0:
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    distance = math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
                    if distance < close:
                        close = distance
                        closest_point = [x, y]
        if close != closest:
            return closest_point


class Wall:
    def __init__(self, start_pos, end_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos
        # Multiple visible ranges for partially hidden walls
        self.visible_ranges = []  # List of (start, end) tuples
        self.current_hits = []  # Store current ray hits this frame
        self.active_rays = set()  # Track which rays are currently hitting this wall
        self.was_hit = False  # Track if wall was hit in current frame
        self.hit = False  # Track if wall was ever hit
        self.flag = False  # i truly hope this helps to remove the rays from the walls

        # Wall vector calculations
        self.vector = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
        self.length = math.sqrt(self.vector[0] ** 2 + self.vector[1] ** 2)
        self.direction = (self.vector[0] / self.length, self.vector[1] / self.length) if self.length > 0 else (0, 0)

        # For tracking extreme points
        self.leftmost_hit = None
        self.rightmost_hit = None

    def get_random_point(self):
        start, end = self.visible_ranges[np.random.choice(len(self.visible_ranges))]
        t = np.random.uniform(0, 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        return x, y

    def update_visible_ranges(self):
        if not self.was_hit:
            # If wall wasn't hit this frame but was hit before
            if self.hit:
                # Check if any rays are still active on this wall
                if not [ray for ray in self.active_rays if ray.last_hit_wall == self]:
                    # No active rays left - reset everything
                    self.visible_ranges = []
                    lidar_walls.remove(self)
                    self.leftmost_hit = None
                    self.rightmost_hit = None
                    self.hit = False
            return

        # Mark that wall was hit at least once
        self.hit = True

        # Convert hits to parameters along the wall
        hit_params = [self.point_to_parameter(hit) for hit in self.current_hits]
        hit_params.sort()

        # Group consecutive hits into visible ranges
        if hit_params:
            # Convert parameters back to points
            self.visible_ranges = [
                (self.parameter_to_point(hit_params[0]), self.parameter_to_point(hit_params[-1]))
            ]
            if self not in lidar_walls:
                lidar_walls.append(self)

            # Update extreme hits
            self.leftmost_hit = self.parameter_to_point(hit_params[0])
            self.rightmost_hit = self.parameter_to_point(hit_params[-1])

        # Reset for next frame
        self.current_hits = []
        self.was_hit = False

    def point_to_parameter(self, point):
        """Convert point to parameter t (0-1) along wall"""
        if self.length == 0:
            return 0
        dx = point[0] - self.start_pos[0]
        dy = point[1] - self.start_pos[1]
        return (dx * self.direction[0] + dy * self.direction[1]) / self.length

    def parameter_to_point(self, t):
        """Convert parameter t to point on wall"""
        x = self.start_pos[0] + t * self.vector[0]
        y = self.start_pos[1] + t * self.vector[1]
        return x, y

    def debug_draw(self):
        if self.visible_ranges:
            arcade.draw_line(*self.visible_ranges[0][0], *self.visible_ranges[0][1], arcade.color.MAGENTA, 2)
        if self.leftmost_hit:
            arcade.draw_point(*self.leftmost_hit, arcade.color.BLUE, 10)
        if self.rightmost_hit:
            arcade.draw_point(*self.rightmost_hit, arcade.color.PURPLE, 10)


class BVHNode:
    def __init__(self, walls):
        self.aabb = self._compute_node_aabb(walls)
        self.walls = walls
        self.left = None
        self.right = None
        self.wall_normals = [self._compute_wall_normal(wall) for wall in walls]
        # Добавляем случайный цвет для визуализации
        self.color = (
            np.random.randint(50, 200),
            np.random.randint(50, 200),
            np.random.randint(50, 200),
            80  # Полупрозрачный
        )

    def _compute_node_aabb(self, walls):
        """Приватный метод: вычисляет AABB только для этого узла"""
        min_x = min(min(wall.start_pos[0], wall.end_pos[0]) for wall in walls)
        max_x = max(max(wall.start_pos[0], wall.end_pos[0]) for wall in walls)
        min_y = min(min(wall.start_pos[1], wall.end_pos[1]) for wall in walls)
        max_y = max(max(wall.start_pos[1], wall.end_pos[1]) for wall in walls)
        return min_x, min_y, max_x, max_y

    def _compute_wall_normal(self, wall):
        """Приватный метод: вычисляет нормаль конкретной стены"""
        (x1, y1), (x2, y2) = wall.start_pos, wall.end_pos
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        normal = (-dy/length, dx/length)
        # Корректировка направления нормали
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        if (mid_x * normal[0] + mid_y * normal[1]) < 0:
            normal = (-normal[0], -normal[1])
        return normal


class Something:
    def __init__(self, cent_x, cent_y, width, height):
        self.x = cent_x
        self.y = cent_y
        self.width = width
        self.height = height
        self.point_list = ((cent_x-width//2, cent_y),
                           (cent_x, cent_y+height//2),
                           (cent_x+width//2, cent_y),
                           (cent_x, cent_y-height//2))
        self.speed = 1
        self.change_x = 0
        self.change_y = 0
        self.enraged = False
        self.sound = arcade.load_sound('rc_asts/close.wav', streaming=True)
        self.sound_player = None
        self.sound_volume = 0
        self.hit_sounds = [arcade.load_sound('rc_asts/hits/flinch (1).mp3', streaming=True),
                           arcade.load_sound('rc_asts/hits/flinch (2).mp3', streaming=True),
                           arcade.load_sound('rc_asts/hits/flinch (3).mp3', streaming=True),
                           arcade.load_sound('rc_asts/hits/flinch (4).mp3', streaming=True),
                           arcade.load_sound('rc_asts/hits/flinch (5).mp3', streaming=True)]
        self.hit_player = None
        self.rage_sound = arcade.load_sound('rc_asts/rage.wav', streaming=True)
        self.rage_player = None

    def move(self, pl_x, pl_y):
        self.x += self.change_x
        self.y += self.change_y
        x_diff = pl_x - self.x
        y_diff = pl_y - self.y
        angle = math.atan2(y_diff, x_diff)
        self.change_x = math.cos(angle) * self.speed
        self.change_y = math.sin(angle) * self.speed
        self.point_list = ((self.x - self.width // 2, self.y),
                           (self.x, self.y + self.height // 2),
                           (self.x + self.width // 2, self.y),
                           (self.x, self.y - self.height // 2))
        if (x_diff ** 2 + y_diff ** 2) <= (self.width*3)**2:
            self.sound_volume = (1 - (x_diff ** 2 + y_diff ** 2) / (self.width * 3) ** 2)*SFX_VOLUME
            if self.sound_player is None:
                self.sound_player = self.sound.play(volume=self.sound_volume, loop=True)
            elif self.sound.is_playing(self.sound_player):
                self.sound_player.volume = self.sound_volume
        else:
            if self.sound_player is not None:
                self.sound_player.volume = 0

    def draw(self):
        arcade.draw_polygon_outline(self.point_list, arcade.color.SPANISH_VIOLET, 3)

    def checkPlayerCollision(self, x, y):
        x1 = abs(x-self.x)
        y1 = abs(y-self.y)
        if (x1/(self.width//2) + y1/(self.height//2)) <= 1:
            return True

    def hitsound(self):
        self.hit_player = self.hit_sounds[np.random.choice(len(self.hit_sounds))].play(volume=SFX_VOLUME/2)

    def random_xy(self):
        one = np.random.randint(0, 2)
        two = np.random.randint(0, 2)
        if (one or not (not one and mx - 600 > area[0][0])) and mx + 600 < area[1][0]:
            self.x = np.random.randint(mx + 600, area[1][0])
        elif not one and mx-600 > area[0][0]:
            self.x = np.random.randint(area[0][0], mx - 600)
        else:
            self.x = area[0][0] # not going to fix this, doesnt really matter that much
        if (two or not (not two and my - 600 > 0)) and my + 600 < area[1][1]:
            self.y = np.random.randint(my + 600, area[1][1])
        elif not one and my-600 > area[0][1]:
            self.y = np.random.randint(area[0][1], my - 600)
        else:
            pass

    def update_rage(self):
        if len(visited_areas) >= area_limit1 and not self.enraged:
            self.enraged = True
            self.speed = 5
            self.rage_player = self.rage_sound.play(volume=SFX_VOLUME)


class EscapeBox:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.sound = arcade.load_sound('rc_asts/exit.mp3', streaming=True)
        self.sound_volume = 0
        self.sound_radius = 600
        self.sound_player = None
        self.len = 25

    def update_sound(self, x, y):
        x_diff = x - self.x
        y_diff = y - self.y
        if (x_diff ** 2 + y_diff ** 2) <= self.sound_radius**2:
            self.sound_volume = (1 - (x_diff ** 2 + y_diff ** 2) / self.sound_radius ** 2)*SFX_VOLUME
            if self.sound_player is None:
                self.sound_player = self.sound.play(volume=self.sound_volume, loop=True)
            elif self.sound.is_playing(self.sound_player):
                self.sound_player.volume = self.sound_volume
        else:
            if self.sound_player is not None:
                self.sound_player.volume = 0

    def stop_sound(self):
        try:
            self.sound.stop(self.sound_player)
            self.sound_player = None
        except:
            self.sound_player = None

    def checkPlayerCollision(self, x, y):
        global win
        if (self.x - self.len // 2 <= x <= self.x + self.len // 2) and (self.y - self.len // 2 <= y <= self.y + self.len// 2):
            win = True

    def draw(self):
        arcade.draw_rect_filled(arcade.rect.XYWH(self.x, self.y, self.len, self.len), (59, 68, 75, 60))


class PlayerSprite:
    def __init__(self):
        self.sprites = [arcade.Sprite('rc_asts/player/up.png'),
                        arcade.Sprite('rc_asts/player/right.png'),
                        arcade.Sprite('rc_asts/player/down.png'),
                        arcade.Sprite('rc_asts/player/left.png')]
        self.sprite_list = arcade.SpriteList()
        self.position = mx, my

    def draw(self):
        self.sprite_list.clear()
        if 45 < curr_angle <= 135:
            sprite = self.sprites[0]
            sprite.position = self.position
            self.sprite_list.append(sprite)
        elif -45 < curr_angle <= 45:
            sprite = self.sprites[1]
            sprite.position = self.position
            self.sprite_list.append(sprite)
        elif -135 < curr_angle <= -45:
            sprite = self.sprites[2]
            sprite.position = self.position
            self.sprite_list.append(sprite)
        elif -180 <= curr_angle <= -135 or 135 <= curr_angle <= 180:
            sprite = self.sprites[3]
            sprite.position = self.position
            self.sprite_list.append(sprite)
        self.sprite_list.draw()

    def update(self):
        self.position = mx, my


creature = Something(0, 0, 100, 50)


def build_bvh(walls, depth=0, max_depth=20):
    """Рекурсивно строит BVH-дерево с автоматическим определением осей разделения."""

    # Базовый случай - создаём лист
    if len(walls) <= 4 or depth >= max_depth:
        return BVHNode(walls)

    # 1. Вычисляем общий AABB для всех стен
    all_min_x = min(min(wall.start_pos[0], wall.end_pos[0]) for wall in walls)
    all_max_x = max(max(wall.start_pos[0], wall.end_pos[0]) for wall in walls)
    all_min_y = min(min(wall.start_pos[1], wall.end_pos[1]) for wall in walls)
    all_max_y = max(max(wall.start_pos[1], wall.end_pos[1]) for wall in walls)

    # 2. Определяем лучшую ось для разделения (x или y)
    dx = all_max_x - all_min_x
    dy = all_max_y - all_min_y
    axis = 0 if dx > dy else 1  # 0 - ось X, 1 - ось Y

    # 3. Сортируем стены по средней точке на выбранной оси
    walls_sorted = sorted(walls, key=lambda wall: (
            (wall.start_pos[axis] + wall.end_pos[axis]) / 2))  # Средняя точка стены

    # 4. Разделяем стены примерно пополам
    mid = len(walls_sorted) // 2
    left_walls = walls_sorted[:mid]
    right_walls = walls_sorted[mid:]

    # 5. Рекурсивно строим левое и правое поддеревья
    node = BVHNode(walls)  # Создаём узел (но пока без детей)
    node.left = build_bvh(left_walls, depth + 1, max_depth)
    node.right = build_bvh(right_walls, depth + 1, max_depth)

    return node


def draw_bvh(node):
    if node is None:
        return
    min_x, min_y, max_x, max_y = map(int, node.aabb)
    arcade.draw_rect_outline(arcade.XYWH(min_x, min_y, max_x - min_x, max_y - min_y, anchor=Vec2(0.0, 0.0)), node.color, 2)
    # Рекурсивно рисуем левую и правую ветви
    draw_bvh(node.left)
    draw_bvh(node.right)


def intersect_bvh_with_counting(node, ray, return_wall=False):
    """Поиск с подсчётом и возвратом стены"""
    global intersect_count
    if node is None:
        return (None, float('inf'), None) if return_wall else (None, float('inf'))

    intersect_count += 1

    if not aabb_intersects_ray(node.aabb, ray):
        return (None, float('inf'), None) if return_wall else (None, float('inf'))

    if node.left is None and node.right is None:
        closest_intersection = None
        min_distance = ray_len
        hit_wall = None

        for i, wall in enumerate(node.walls):
            intersect_count += 1
            intersection = ray.checkCollision(wall)
            if intersection:
                wall.flag = True
                dist = math.hypot(intersection[0] - ray.x, intersection[1] - ray.y)
                if dist < min_distance:
                    min_distance = dist
                    closest_intersection = intersection
                    hit_wall = wall

        if return_wall:
            return closest_intersection, min_distance, hit_wall
        return closest_intersection, min_distance

    left_result = intersect_bvh_with_counting(node.left, ray, return_wall)
    right_result = intersect_bvh_with_counting(node.right, ray, return_wall)

    if return_wall:
        left_intersection, left_dist, left_wall = left_result
        right_intersection, right_dist, right_wall = right_result
        if left_dist < right_dist:
            return left_intersection, left_dist, left_wall
        return right_intersection, right_dist, right_wall
    else:
        left_intersection, left_dist = left_result
        right_intersection, right_dist = right_result
        if left_dist < right_dist:
            return left_intersection, left_dist
        return right_intersection, right_dist


def aabb_intersects_ray(aabb, ray):
    """Проверяет, пересекает ли луч ограничивающий объем (AABB)"""
    min_x, min_y, max_x, max_y = aabb

    if (ray.x < min_x and ray.end_x < min_x) or (ray.x > max_x and ray.end_x > max_x):
        return False
    if (ray.y < min_y and ray.end_y < min_y) or (ray.y > max_y and ray.end_y > max_y):
        return False
    return True


def point_on_segment(p, a, b):
    """Проверяет, лежит ли точка p на отрезке ab."""
    px, py = p
    ax, ay = a
    bx, by = b
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > 1e-6:
        return False
    min_x = min(ax, bx)
    max_x = max(ax, bx)
    min_y = min(ay, by)
    max_y = max(ay, by)
    return (min_x <= px <= max_x) and (min_y <= py <= max_y)


# rays init
start = 0
end = 90
for i in range(start, end, int(90 / NUM_RAYS)):
    rays.append(Ray(mx, my, math.radians(i)))


def drawRays(rays, walls):
    global lastClosestPoint, intersect_count, precreaturedots
    raysbuff = []
    raydotsbuff = []
    precreaturedots = []
    for i, ray in enumerate(rays):
        closest = ray_len
        closest_point = None
        closest_wall = None
        if bvh_flag:
            closest_point, _, closest_wall = intersect_bvh_with_counting(bvh_root, ray, True)
            if closest_point is not None:
                distance = math.sqrt((ray.x - closest_point[0]) ** 2 + (ray.y - closest_point[1]) ** 2)
                if distance < closest:
                    closest = distance
        else:
            intersect_count += 1
            for wall in walls:
                intersect = ray.checkCollision(wall)
                intersect_count += 1
                if intersect:
                    wall.flag = True
                    distance = math.sqrt((ray.x - intersect[0]) ** 2 + (ray.y - intersect[1]) ** 2)
                    if distance < closest:
                        closest = distance
                        closest_point = intersect
                        closest_wall = wall

        kreatur = ray.checkCreatureCollision(creature, closest)
        if kreatur is not None:
            precreaturedots.append(kreatur)

        if closest_point is None:
            closest_point = [ray.x + closest * math.cos(ray.angle), ray.y + closest * math.sin(ray.angle)]
            ray.last_hit_wall = None
        if closest_wall:
            closest_wall.current_hits.append(closest_point)
            closest_wall.was_hit = True

            if ray.last_hit_wall != closest_wall:
                if ray.last_hit_wall:
                    ray.last_hit_wall.active_rays.discard(ray)
                closest_wall.active_rays.add(ray)
                ray.last_hit_wall = closest_wall
            if i == 0:
                rays[0] = (closest_wall, closest_point)
            elif i == len(rays) - 1:
                rays[-1] = (closest_wall, closest_point)

        if i == 0 or i == len(rays) - 1:
            arcade.draw_line(ray.x, ray.y, closest_point[0], closest_point[1],
                             arcade.color.GREEN)
        elif ray_flag:
            raysbuff.append((ray.x, ray.y))
            raysbuff.append((closest_point[0], closest_point[1]))
        raydotsbuff.append([closest_point[0], closest_point[1]])

    for wall in walls:
        wall.update_visible_ranges()
        if wall.flag:
            wall.was_hit = False
            wall.active_rays.clear()

    if ray_flag and raysbuff:
        arcade.draw_lines(raysbuff, arcade.color.WHITE)
    if not DEBUG_FLAG:
        arcade.draw_points(raydotsbuff, arcade.color.GREEN, size=3)


def getWallDot():
    wall = lidar_walls[np.random.choice(len(lidar_walls))]
    point = wall.get_random_point()
    walldots.append(point)


def drawWallDots():
    arcade.draw_points(walldots, arcade.color.CARIBBEAN_GREEN, 4)
    if len(walldots) > MAX_DOTS:
        walldots.remove(walldots[0])


def getKreaturaDot():
    point = precreaturedots[np.random.choice(len(precreaturedots))]
    creaturedots.append(point)


def drawKreaturaDots():
    global bvh_root
    arcade.draw_points(creaturedots, arcade.color.RED, 4)
    if len(creaturedots) > 20:
        creaturedots.clear()
        drawwalls.clear()
        creature.enraged = False
        creature.speed = 1
        walls.clear()
        walldots.clear()
        lidar_walls.clear()
        visited_areas.clear()
        for void in escapes:
            void.stop_sound()
        escapes.clear()
        generateWalls()
        bvh_root = build_bvh(walls)
        creature.hitsound()
        creature.random_xy()


def generateWalls(x1=area[0][0], x2=area[1][0], y1=area[0][1], y2=area[1][1], flag=True):\
    # drawwalls.clear()
    # if flag:
    #     walls.append(Wall((0, 0), (window.width, 0)))
    #     walls.append(Wall((0, 0), (0, window.height)))
    #     walls.append(Wall((window.width, 0), (window.width, window.height)))
    #     walls.append(Wall((0, window.height), (window.width, window.height)))

    for i in range(NUM_WALLS):
        start_x = np.random.randint(x1, x2)
        start_y = np.random.randint(y1, y2)
        end_x = np.random.randint(x1, x2)
        end_y = np.random.randint(y1, y2)
        walls.append(Wall((start_x, start_y), (end_x, end_y)))
        drawwalls.append((start_x, start_y))
        drawwalls.append((end_x, end_y))

    if np.random.randint(0, 10) == 1 and can_escape:
        print('a way out has appeared.')
        x = np.random.randint(x1, x2)
        y = np.random.randint(y1, y2)
        escapes.append(EscapeBox(x, y))
    visited_areas.append(area)


def changeRays():
    global start, end, up1, down1
    if left or right or up or down or up1 or down1:
        if left:
            start += 1
            end += 1
        if right:
            start -= 1
            end -= 1
        if up and (end - start != 360):
            start -= 1
            end += 1
        if down and (end - start != 2):
            start += 1
            end -= 1
        if up1 and (end - start != 360):
            start -= 1
            end += 1
            up1 = False
        if down1 and (end - start != 2):
            start += 1
            end -= 18
            down1 = False
        rays.clear()
        for i in range(start, end, int(90 / NUM_RAYS)):
            rays.append(Ray(mx, my, math.radians(i)))


def checkPlayerCollision(movement, speed):
    """
    :param movement: left, right, up or down
    :return: bool
    """
    global mx, my
    player_speed = speed
    for wall in walls:
        x1, y1 = wall.start_pos
        x2, y2 = wall.end_pos
        x3, y3 = mx, my
        if movement == 'up':
            x4, y4 = mx, my + player_speed  # i aint making another function just for these ok, it works and thats good
            try:
                t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                u = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return True
                pass
            except:
                pass
        elif movement == 'down':
            x3, y3 = mx, my - player_speed
            x4, y4 = mx, my
            try:
                t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                u = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return True
                pass
            except:
                pass
        elif movement == 'right':
            x4, y4 = mx + player_speed, my
            try:
                t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                u = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return True
                pass
            except:
                pass
        elif movement == 'left':
            x3, y3 = mx - player_speed, my
            x4, y4 = mx, my
            try:
                t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                u = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / ((y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1))
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return True
                pass
            except:
                pass
    return False


def move_player():
    global mup, mdown, mleft, mright, mx, my, area, bvh_root, stamina
    if mup or mdown or mleft or mright:
        if mrun and 200 <= stamina <= 10000:
            stamina -= 200
            player_speed = 2*START_PLAYER_SPEED
        else:
            player_speed = START_PLAYER_SPEED
        if mup:
            if not checkPlayerCollision('up', player_speed):
                my += player_speed
        if mdown:
            if not checkPlayerCollision('down', player_speed):
                my -= player_speed
        if mleft:
            if not checkPlayerCollision('left', player_speed):
                mx -= player_speed
        if mright:
            if not checkPlayerCollision('right', player_speed):
                mx += player_speed
        if mx <= area[0][0]:
            area = [(area[0][0] - window.width, area[0][1]), (area[0][0], area[1][1])]
            if area not in visited_areas:
                generateWalls(area[0][0], area[1][0], area[0][1], area[1][1])
                bvh_root = build_bvh(walls)
        elif mx >= area[1][0]:
            area = [(area[1][0], area[0][1]), (area[1][0] + window.width, area[1][1])]
            if area not in visited_areas:
                generateWalls(area[0][0], area[1][0], area[0][1], area[1][1])
                bvh_root = build_bvh(walls)
        if my <= area[0][1]:
            area = [(area[0][0], area[0][1] - window.height), (area[1][0], area[0][1])]
            if area not in visited_areas:
                generateWalls(area[0][0], area[1][0], area[0][1], area[1][1])
                bvh_root = build_bvh(walls)
        elif my >= area[1][1]:
            area = [(area[0][0], area[1][1]), (area[1][0], area[1][1] + window.height)]
            if area not in visited_areas:
                generateWalls(area[0][0], area[1][0], area[0][1], area[1][1])
                bvh_root = build_bvh(walls)
        for ray in rays:
            ray.update(mx, my)


def update_angle(x, y):
    x_angle = x - window.center_x
    y_angle = y - window.center_y
    angle = -math.atan2(-y_angle, x_angle)
    return round(math.degrees(angle))


generateWalls()
bvh_root = build_bvh(walls)


class GameView(arcade.View):

    def __init__(self):
        super().__init__()
        self.background_color = arcade.color.BLACK
        self.fps = 0.0
        self.fpstext = arcade.Text(x=5.0, y=self.height - 14, text=f'FPS: {self.fps}')
        self.inttext = arcade.Text(x=5.0, y=14.0, text=f'Intersection: {intersect_count}')
        self.gotext = arcade.Text(x=window.center_x, y=window.center_y, text="You are dead :(\nPress Enter to restart", color=arcade.color.WHITE, anchor_x='center', anchor_y='center', width=200, multiline=True)
        self.wintext = arcade.Text(x=window.center_x, y=window.center_y, text="You escaped the void!\n...or did you?",
                                  anchor_x='center', anchor_y='center', width=200, multiline=True, color=arcade.color.BLACK)
        self.ragetext = arcade.Text(x=window.center_x, y=20, text="Something is enraged. Beware.", color=arcade.color.RED, anchor_x='center', width=200)
        self.staminatext = arcade.Text(x=130, y=60, text="Stamina", anchor_x="center", anchor_y='center', width=200)
        self.clocker = arcade.clock.Clock()
        self.recharge_flag = True
        self.textflag = False
        self.bgmusic = None
        self.draw_outlines = False
        self.player = PlayerSprite()
        self.audio = arcade.load_sound('rc_asts/spooky_mita.mp3', streaming=True)
        self.exit_snd = arcade.load_sound('rc_asts/leave.mp3', streaming=True)
        self.void_snd = arcade.load_sound('rc_asts/void.mp3', streaming=True)
        self.death_snd = arcade.load_sound('rc_asts/death.wav', streaming=True)
        self.sound_playing = False
        self.death_gif = gifSprite('rc_asts/died.gif', True)
        self.died = gifSprite('rc_asts/died.gif', True, 0.5)
        self.died.position = window.center_x, window.center_y + 25
        self.died.size = window.width, window.height
        self.gif_list = arcade.SpriteList()
        self.gif_launched = False
        self.exit_player = None
        self.camera_sprites = arcade.Camera2D()
        self.camera_gui = arcade.Camera2D()
        self.wht_delta = 0
        self.bgflag = True
        self.game_over1 = False
        self.game_over2 = False

    def scroll_to_player(self):
        self.camera_sprites.position = arcade.math.lerp_2d(self.camera_sprites.position, (mx, my), 1)

    def on_show_view(self):
        if self.bgmusic is None:
            self.bgmusic = self.audio.play(volume=MUSIC_VOLUME, loop=True)

    def on_draw(self):
        global intersect_count
        self.clear()
        if not self.game_over2 and not win:
            intersect_count = 0

            self.camera_sprites.use()
            if self.draw_outlines:
                arcade.draw_lines(drawwalls, arcade.color.WHITE, 3)
                creature.draw()
            if escapes:
                for void in escapes:
                    void.draw()
            if DEBUG_FLAG:
                for wall in walls:
                    wall.debug_draw()
            drawRays([ray for ray in rays], [wall for wall in walls])
            if walldots:
                drawWallDots()
            if creaturedots:
                drawKreaturaDots()
            if drawbvh and bvh_flag:
                draw_bvh(bvh_root)
            self.player.draw()

            self.camera_gui.use()
            self.staminatext.draw()
            arcade.draw_rect_outline(arcade.rect.XYWH(130, 30, 200, 30), arcade.color.GRAY)
            arcade.draw_rect_filled(arcade.rect.XYWH(130, 30, 2*(stamina//100), 30), arcade.color.BONDI_BLUE)
            if self.textflag:
                self.fpstext.draw()
                self.inttext.draw()
            if creature.enraged:
                self.ragetext.draw()
        if self.game_over1:
            arcade.draw_rect_filled(arcade.rect.XYWH(window.center_x, window.center_y, window.width, window.height), (0, 0, 0, self.wht_delta))
            if self.wht_delta == 255:
                self.gotext.draw()
            self.gif_list.draw()
        if win:
            if self.exit_player is None:
                self.exit_player = self.exit_snd.play(volume=SFX_VOLUME)
            if self.wht_delta != 255:
                self.camera_sprites.use()
                if escapes:
                    for void in escapes:
                        void.draw()
                drawRays([ray for ray in rays], [wall for wall in walls])
                if walldots:
                    drawWallDots()
                if creaturedots:
                    drawKreaturaDots()
                self.player.draw()
            arcade.draw_rect_filled(arcade.rect.XYWH(mx, my, window.width, window.height), (128, 128, 128, self.wht_delta))
            if self.wht_delta == 255:
                self.wintext.x, self.wintext.y = mx, my
                self.wintext.draw()

    def on_update(self, delta_time: float):
        global stamina, can_escape
        if not (win or self.game_over1 or self.game_over2):
            changeRays()
            if not self.recharge_flag:
                self.clocker.tick(delta_time)
                if self.clocker.ticks % dot_speed == 0:
                    self.recharge_flag = True
            if lidar_flag and self.recharge_flag:
                if lidar_walls:
                    getWallDot()
                if precreaturedots:
                    getKreaturaDot()
                self.recharge_flag = False

            self.fps = round(1 / delta_time, 2)
            if stamina != 10000:
                stamina += 20
                if stamina > 10000:
                    stamina = 10000
            if len(visited_areas) >= 6 and not can_escape:
                can_escape = True
            move_player()
            if escapes:
                for void in escapes:
                    void.update_sound(mx, my)
                    void.checkPlayerCollision(mx, my)
            creature.update_rage()
            creature.move(mx, my)
            self.game_over1 = creature.checkPlayerCollision(mx, my)
            self.player.update()
            self.scroll_to_player()
            self.inttext.text = f'Intersection: {intersect_count}'
        elif win:
            creature.sound_player.volume = 0
            if self.bgmusic is not None:
                if self.bgflag:
                    self.audio.stop(self.bgmusic)
                    self.bgmusic = None
                    self.bgflag = False
            else:
                if self.wht_delta == 255:
                    self.bgmusic = self.void_snd.play(volume=MUSIC_VOLUME, loop=True)
            if escapes:
                for void in escapes:
                    void.stop_sound()
                escapes.clear()
            if self.wht_delta != 255:
                self.wht_delta += 2.5
                if self.wht_delta > 255:
                    self.wht_delta = 255
        elif self.game_over1:
            creature.sound_player.volume = 0
            if not self.sound_playing:
                try:
                    self.death_snd.play(volume=SFX_VOLUME)
                except:
                    pass
                self.sound_playing = True
            if escapes:
                for void in escapes:
                    void.stop_sound()
                escapes.clear()
            if not self.gif_launched:
                if not self.gif_list:
                    self.gif_list.append(self.died)
                else:
                    self.died.current_texture = 0
                    self.died.time_elapsed = 0
                self.gif_launched = True
            self.gif_list.update(delta_time)
            if self.wht_delta != 255:
                self.wht_delta += 2.5
                if self.wht_delta > 255:
                    self.wht_delta = 255
            if not self.gif_list and not self.game_over2:
                self.game_over2 = True

        self.fpstext.text = f'FPS: {self.fps}'

    def on_key_press(self, key, key_modifiers):
        # I KNOW THIS IS TOO MANY GLOBALS LOL, i wont fix this as of now cause THIS IS JUST A TECH DEMO GOD DAMMIT
        global left, right, up, down, stamina, lidar_flag, ray_flag, ray_mode, DEBUG_FLAG, bvh_root, bvh_flag, drawbvh, mup, mdown, mleft, mright, mrun, mx, my, start, end, ray_len, dot_speed, area
        if not (self.game_over1 or self.game_over2):
            if key == arcade.key.SPACE:
                lidar_walls.clear()
                visited_areas.clear()
                walls.clear()
                drawwalls.clear()
                escapes.clear()
                creature.enraged = False
                generateWalls()
                bvh_root = build_bvh(walls)
            if key == arcade.key.J:
                bvh_flag = not bvh_flag
            if key == arcade.key.N:
                drawbvh = not drawbvh
            if key == arcade.key.H:
                DEBUG_FLAG = not DEBUG_FLAG
            if key == arcade.key.B:
                self.draw_outlines = not self.draw_outlines
            if key == arcade.key.LEFT:
                left = True
            if key == arcade.key.RIGHT:
                right = True
            # if key == arcade.key.UP:
            #     up = True
            # if key == arcade.key.DOWN:
            #     down = True
            if key == arcade.key.W:
                mup = True
            if key == arcade.key.A:
                mleft = True
            if key == arcade.key.S:
                mdown = True
            if key == arcade.key.D:
                mright = True
            if key == arcade.key.LSHIFT:
                mrun = True
            if key == arcade.key.F:
                lidar_flag = True
            if key == arcade.key.K:
                walldots.clear()
                creaturedots.clear()
            if key == arcade.key.R:
                ray_flag = not ray_flag
            if key == arcade.key.M:
                ray_mode = not ray_mode
                if ray_mode:
                    start = curr_angle - 45
                    end = curr_angle + 45
                    ray_len = MIN_RAY_LEN
                    dot_speed = BASE_DOT_SPEED
                else:
                    start = curr_angle - 12
                    end = curr_angle + 13
                    ray_len = 2*MIN_RAY_LEN
                    dot_speed = 2
                rays.clear()
                for i in range(start, end, int(90 / NUM_RAYS)):
                    rays.append(Ray(mx, my, math.radians(i)))
            if key == arcade.key.L:
                self.textflag = not self.textflag
        else:
            if key == arcade.key.ENTER:
                creaturedots.clear()
                walldots.clear()
                lidar_walls.clear()
                visited_areas.clear()
                walls.clear()
                drawwalls.clear()
                escapes.clear()
                area = [(0, 0), (window.width, window.height)]
                mx, my = 600, 300
                generateWalls()
                creature.random_xy()
                self.gif_launched = False
                self.sound_playing = False
                bvh_root = build_bvh(walls)
                mup, mdown, mleft, mright = False, False, False, False
                lidar_flag = False
                stamina = 10000
                self.wht_delta = 0
                self.game_over1 = False
                self.game_over2 = False
                for ray in rays:
                    ray.update(mx, my)
        if key == arcade.key.ESCAPE:
            window.close()

    def on_key_release(self, key, key_modifiers):
        global left, right, up, down, lidar_flag, mup, mdown, mleft, mright, mrun
        if not self.game_over1:
            if key == arcade.key.LEFT:
                left = False
            if key == arcade.key.RIGHT:
                right = False
            # if key == arcade.key.UP:
            #     up = False
            # if key == arcade.key.DOWN:
            #     down = False
            if key == arcade.key.W:
                mup = False
            if key == arcade.key.A:
                mleft = False
            if key == arcade.key.S:
                mdown = False
            if key == arcade.key.D:
                mright = False
            if key == arcade.key.LSHIFT:
                mrun = False
            if key == arcade.key.F:
                lidar_flag = False

    def on_mouse_motion(self, x, y, delta_x, delta_y):
        global curr_angle, start, end
        angle = update_angle(x, y)
        if angle != curr_angle:
            if ray_mode:
                start = angle - 45
                end = angle + 45
            else:
                start = angle - 12
                end = angle + 13
            curr_angle = angle
            rays.clear()
            for i in range(start, end, int(90 / NUM_RAYS)):
                rays.append(Ray(mx, my, math.radians(i)))

    def on_mouse_press(self, x, y, button, modifiers):
        global lidar_flag
        if button == arcade.MOUSE_BUTTON_LEFT:
            lidar_flag = True

    def on_mouse_release(self, x, y, button, modifiers):
        global lidar_flag
        if button == arcade.MOUSE_BUTTON_LEFT:
            lidar_flag = False


def main():
    game = GameView()
    window.show_view(game)
    arcade.run()


if __name__ == "__main__":
    main()
