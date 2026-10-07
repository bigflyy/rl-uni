"""Создание карты Павлодара. Запуск: python build_map.py."""

# Работа с путями, поиском по клеткам и координатами карты.
from pathlib import Path
from collections import deque
import math
import xml.etree.ElementTree as ET

# Загрузка OSM, массивы клеток и рисование изображений.
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from PIL import Image as PILImage, ImageDraw, ImageFont

# Сохраняем результат рядом с этим скриптом, в папке data.
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Центр участка, половина стороны в метрах и число клеток по стороне.
CENTER_LATITUDE = 52.2873
CENTER_LONGITUDE = 76.9674
MAP_HALF_SIDE_METERS = 950
MAP_CELLS_PER_SIDE = 160

# Типы дорог, присутствующие на выбранном участке; дворовые проезды не включаем.
ROAD_TYPES = {"primary", "secondary", "tertiary", "residential", "primary_link"}
NEIGHBOR_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Старты и цель заданы явно: (строка, столбец), отсчёт с нуля.
START_CELLS = [(8, 14), (100, 125), (154, 9)]
GOAL_CELL = (24, 159)

# Число источников машин и seed для их выбора среди дорожных клеток.
TRAFFIC_SOURCE_COUNT = 6
TRAFFIC_SOURCE_SEED = 2026


# Возвращаем соседние клетки дороги внутри карты.
def road_neighbors(cell, road_grid):
    row, column = cell
    neighbors = []

    # Проверяем четыре направления и пропускаем препятствия и край карты.
    for row_step, column_step in NEIGHBOR_OFFSETS:
        next_row = row + row_step
        next_column = column + column_step
        if 0 <= next_row < len(road_grid) and 0 <= next_column < len(road_grid[0]):
            if road_grid[next_row, next_column]:
                neighbors.append((next_row, next_column))
    return neighbors


# Находим все клетки дороги, до которых можно доехать от заданной клетки.
def connected_road_cells(road_grid, start_cell):

    # Начинаем поиск с выбранной клетки.
    found_cells = {start_cell}
    queue = deque([start_cell])

    # Обходим клетки по очереди: сначала ближайшие, затем более далёкие.
    while queue:
        cell = queue.popleft()

        # Проверяем соседей текущей клетки и добавляем ещё не посещённые.
        for next_cell in road_neighbors(cell, road_grid):
            # Каждую клетку добавляем в очередь только один раз.
            if next_cell not in found_cells:
                found_cells.add(next_cell)
                queue.append(next_cell)

    return found_cells


# Рисуем отрезок дороги клетками без диагональных разрывов.
def rasterize_road_segment(road_grid, start_cell, end_cell):

    # Разбираем координаты концов отрезка и ставим текущую клетку в начало.
    start_row, start_column = start_cell
    end_row, end_column = end_cell
    row, column = start_cell

    # Сколько всего шагов нужно сделать по строкам и по столбцам.
    total_row_steps = abs(end_row - start_row)
    total_column_steps = abs(end_column - start_column)
    road_grid[row, column] = True

    # Добавляем по одной клетке дороги, пока не дойдём до конца отрезка.
    while (row, column) != (end_row, end_column):
        # Если нужная строка уже достигнута, остаются только шаги по столбцам.
        if row == end_row:
            move_vertically = False
        # Если нужный столбец уже достигнут, остаются только шаги по строкам.
        elif column == end_column:
            move_vertically = True
        else:
            # Доля пройденных шагов по каждой координате: 0.5 означает половину.
            # Здесь оба расстояния больше нуля, поэтому можно делить без max(1, ...).
            row_progress = abs(row - start_row) / total_row_steps
            column_progress = abs(column - start_column) / total_column_steps

            # Двигаемся по координате с меньшей пройденной долей, чтобы следовать отрезку.
            # При равных долях сначала меняем строку.
            move_vertically = row_progress <= column_progress

        # За один шаг меняем только одну координату: строку или столбец.
        if move_vertically:
            if row < end_row:
                row += 1
            else:
                row -= 1
        else:
            if column < end_column:
                column += 1
            else:
                column -= 1

        # Отмечаем новую клетку как дорогу.
        road_grid[row, column] = True


# Собираем подложку из частей карты OSM и сохраняем одной картинкой.
def save_basemap(bounds_lon_lat, data_dir):
    # Настройки подложки. Тайл - квадратная картинка OSM размером 256 x 256 пикселей.
    zoom_level = 15
    tile_size = 256

    # Переводим координаты города в положение на сетке тайлов.
    def tile_position(longitude, latitude):
        tiles_per_axis = 2 ** zoom_level

        # Долгота задаёт положение слева направо.
        tile_x = (longitude + 180) / 360 * tiles_per_axis

        # Для широты используем стандартную формулу проекции карты Web Mercator.
        latitude_radians = math.radians(latitude)
        projected_latitude = math.asinh(math.tan(latitude_radians))
        tile_y = (1 - projected_latitude / math.pi) / 2 * tiles_per_axis
        return tile_x, tile_y

    # Находим верхний левый и нижний правый углы участка в координатах тайлов.
    min_longitude, min_latitude, max_longitude, max_latitude = bounds_lon_lat
    left, top = tile_position(min_longitude, max_latitude)
    right, bottom = tile_position(max_longitude, min_latitude)

    # Округляем вниз: дробная часть показывает положение внутри тайла.
    first_tile_x = math.floor(left)
    first_tile_y = math.floor(top)
    last_tile_x = math.floor(right)
    last_tile_y = math.floor(bottom)

    # Здесь храним скачанные тайлы, чтобы не загружать их при каждом запуске.
    tile_cache_dir = data_dir / "tile_cache"
    tile_cache_dir.mkdir(exist_ok=True)

    # Создаём белую картинку, на которую поместятся все нужные тайлы.
    mosaic_width = (last_tile_x - first_tile_x + 1) * tile_size
    mosaic_height = (last_tile_y - first_tile_y + 1) * tile_size
    tile_mosaic = PILImage.new("RGB", (mosaic_width, mosaic_height), "white")

    # Перебираем столбцы тайлов, покрывающих наш участок.
    for tile_x in range(first_tile_x, last_tile_x + 1):
        # В каждом столбце перебираем тайлы сверху вниз.
        for tile_y in range(first_tile_y, last_tile_y + 1):
            tile_path = tile_cache_dir / f"{zoom_level}_{tile_x}_{tile_y}.png"

            # Скачиваем тайл, только если его ещё нет в папке.
            if not tile_path.exists():
                response = requests.get(
                    f"https://tile.openstreetmap.org/{zoom_level}/{tile_x}/{tile_y}.png",
                    headers={"User-Agent": "rl-uni-coursework-map/1.0 (student project)"}, timeout=30)
                # При ошибке сервера останавливаем загрузку, чтобы не сохранить неверные данные.
                response.raise_for_status()
                tile_path.write_bytes(response.content)

            # Вставляем тайл на его место в общей картинке.
            tile_image = PILImage.open(tile_path).convert("RGB")
            paste_left = (tile_x - first_tile_x) * tile_size
            paste_top = (tile_y - first_tile_y) * tile_size
            tile_mosaic.paste(tile_image, (paste_left, paste_top))

    # Переводим точные границы участка из координат тайлов в пиксели картинки.
    crop_left = round((left - first_tile_x) * tile_size)
    crop_top = round((top - first_tile_y) * tile_size)
    crop_right = round((right - first_tile_x) * tile_size)
    crop_bottom = round((bottom - first_tile_y) * tile_size)
    cropped_map = tile_mosaic.crop((crop_left, crop_top, crop_right, crop_bottom))

    # Сохраняем готовую подложку для карты и анимаций.
    cropped_map.save(data_dir / "source_osm_map.png")


# Читаем линии дорог из OSM и превращаем их в связанную сетку клеток.
def create_road_grid(osm_xml_path, bounds_lon_lat, cells_per_side):

    # В OSM узел (node) - точка с номером id и координатами lat, lon.
    osm_root = ET.parse(osm_xml_path).getroot()
    point_coordinates = {}
    # findall("node") находит все такие точки; сохраняем координаты по их номерам.
    for point in osm_root.findall("node"):
        point_id = point.attrib["id"]
        longitude = float(point.attrib["lon"])
        latitude = float(point.attrib["lat"])
        point_coordinates[point_id] = (longitude, latitude)

    road_paths = []
    # findall("way") берёт линии OSM; оставляем фрагменты нужных типов дорог.
    for osm_way in osm_root.findall("way"):
        road_type = None
        # В <tag k="highway" v="primary"> k - название свойства, v - значение.
        for property_tag in osm_way.findall("tag"):
            if property_tag.attrib["k"] == "highway":
                road_type = property_tag.attrib["v"]
        if road_type not in ROAD_TYPES:
            continue

        road_path = []
        # <nd ref="123"> ссылается на узел с id="123"; добавляем его координаты.
        for point_reference in osm_way.findall("nd"):
            point_id = point_reference.attrib["ref"]
            if point_id in point_coordinates:
                road_path.append(point_coordinates[point_id])
        if len(road_path) >= 2:
            road_paths.append(road_path)

    # Даём понятные названия четырём границам: запад, юг, восток, север.
    min_longitude, min_latitude, max_longitude, max_latitude = bounds_lon_lat

    # Начинаем с пустой сетки: дорога появится только после рисования отрезков.
    road_grid = np.zeros((cells_per_side, cells_per_side), dtype=bool)

    # Переводим географическую точку в строку и столбец клеточной карты.
    def to_grid_cell(point):
        # Север карты сверху: при уменьшении широты номер строки растёт.
        longitude, latitude = point

        # Находим долю пути от верхнего и левого края участка.
        row_fraction = (max_latitude - latitude) / (max_latitude - min_latitude)
        column_fraction = (longitude - min_longitude) / (max_longitude - min_longitude)

        # При 160 клетках их номера идут от 0 до 159, поэтому вычитаем единицу.
        row = round(row_fraction * (cells_per_side - 1))
        column = round(column_fraction * (cells_per_side - 1))

        # Конец линии OSM может лежать за границей участка; ставим его на край сетки.
        if row < 0:
            row = 0
        if row >= cells_per_side:
            row = cells_per_side - 1
        if column < 0:
            column = 0
        if column >= cells_per_side:
            column = cells_per_side - 1
        return row, column

    # Обрабатываем каждый выбранный фрагмент дороги.
    for road_path in road_paths:
        # Соединяем соседние точки фрагмента дорожными клетками.
        for first_point, second_point in zip(road_path, road_path[1:]):
            first_longitude, first_latitude = first_point
            second_longitude, second_latitude = second_point
            # Пропускаем отрезки, которые целиком лежат за одной из границ участка.
            if (max(first_longitude, second_longitude) < min_longitude
                    or min(first_longitude, second_longitude) > max_longitude
                    or max(first_latitude, second_latitude) < min_latitude
                    or min(first_latitude, second_latitude) > max_latitude):
                continue
            rasterize_road_segment(road_grid, to_grid_cell(first_point), to_grid_cell(second_point))
    # Отбрасываем отдельные фрагменты, до которых курьер не может доехать.
    largest_component = set()
    unvisited_road_cells = set()

    # np.argwhere находит координаты клеток, где road_grid равно True.
    for row, column in np.argwhere(road_grid):
        unvisited_road_cells.add((row, column))

    # Находим все отдельные связные участки дорог, каждый раз начиная с непосещённой клетки.
    while unvisited_road_cells:
        connected_cells = connected_road_cells(road_grid, min(unvisited_road_cells))
        if len(connected_cells) > len(largest_component):
            largest_component = connected_cells
        unvisited_road_cells.difference_update(connected_cells)

    # Убираем из сетки все дороги, в том числе отдельные фрагменты.
    road_grid.fill(False)

    # Возвращаем только клетки самого большого связного участка.
    for road_cell in largest_component:
        road_grid[road_cell] = True

    return road_grid, road_paths


# Сохраняем исходные дороги, клеточную карту и карту с подложкой OSM.
def save_map_images(road_grid, road_paths, bounds_lon_lat, start_cells, goal_cell,
                    traffic_sources, center_latitude, data_dir):

    # Создаём рисунок: дороги показаны линиями через точки с координатами OSM.
    cells_per_side = len(road_grid)
    min_longitude, min_latitude, max_longitude, max_latitude = bounds_lon_lat
    figure, axis = plt.subplots(figsize=(7,7))

    # Для каждого фрагмента дороги проводим линию: долгота по X, широта по Y.
    for road_path in road_paths:
        longitudes = []
        latitudes = []
        # Собираем долготы и широты точек в два списка для axis.plot.
        for longitude, latitude in road_path:
            longitudes.append(longitude)
            latitudes.append(latitude)
        axis.plot(longitudes, latitudes, color="#324e68", linewidth=1)

    # Ограничиваем рисунок нашим участком и корректируем соотношение широты и долготы.
    axis.set_xlim(min_longitude, max_longitude)
    axis.set_ylim(min_latitude, max_latitude)
    axis.set_aspect(1/math.cos(math.radians(center_latitude)))
    axis.set_title("Дороги центра Павлодара (OpenStreetMap)")

    # Сохраняем рисунок дорог по координатам и закрываем его.
    figure.savefig(data_dir/"source_roads.png",dpi=160,bbox_inches="tight")
    plt.close(figure)

    # Светлые клетки лежат вне выбранных дорог; тёмные клетки доступны курьеру.
    grid_pixels = np.full((cells_per_side, cells_per_side, 3), (248, 248, 246), dtype="uint8")
    grid_pixels[road_grid] = (42, 71, 89)

    # Собираем цвет и подпись для каждого старта, цели и источника машин.
    map_markers = [(point, (30, 155, 75), f"С{i}") for i, point in enumerate(start_cells, 1)]
    map_markers += [(goal_cell, (233, 111, 31), "Ц")]
    map_markers += [(point, (54, 103, 220), f"Т{i}") for i, point in enumerate(traffic_sources, 1)]

    # Закрашиваем ровно одну клетку для каждой метки.
    for point, color, _ in map_markers:
        grid_pixels[point] = color
    figure, axis = plt.subplots(figsize=(9,9))

    # Показываем сетку без сглаживания, чтобы границы клеток оставались чёткими.
    axis.imshow(grid_pixels, interpolation="nearest")

    # Берём координаты и подпись метки; её цвет здесь не нужен.
    for (row, column), _, label in map_markers:
        # Обычно ставим подпись на две клетки правее метки.
        label_offset = 2
        # Подписи стартов ставим слева, если там достаточно места.
        if (row, column) in start_cells and column >= 10:
            label_offset = -9
        # У правого края переносим любую подпись влево, чтобы она не вышла за карту.
        if column > cells_per_side - 15:
            label_offset = -10
        # Рисуем текст рядом с клеткой на белом фоне для читаемости.
        axis.text(column+label_offset, row-1, label, fontsize=8, weight="bold", color="black",
                bbox={"facecolor":"white", "alpha":.8, "edgecolor":"none", "pad":.5})

    # Добавляем название и легенду цветов.
    axis.set_title(f"Дорожная сетка {cells_per_side} x {cells_per_side}: цветная метка = одна клетка")
    axis.legend(handles=[Patch(color="#2a4759", label="Дорога"), Patch(color="#1e9b4b", label="Старт"),
                       Patch(color="#e96f1f", label="Цель"), Patch(color="#3667dc", label="Источник машин")],
              loc="lower center", fontsize=8, ncol=4)

    # Сохраняем клеточную карту отдельной картинкой.
    figure.savefig(data_dir/"grid_map.png",dpi=160,bbox_inches="tight")
    plt.close(figure)

    # Открываем готовую подложку OSM.
    basemap_path = data_dir / "source_osm_map.png"
    basemap = PILImage.open(basemap_path).convert("RGB")
    width, height = basemap.size

    # Для дороги ставим непрозрачность 120 из 255, вне дороги - 0.
    opacity_cells = road_grid.astype("uint8") * 120

    # Растягиваем маску 160 x 160 до размера подложки без размытия клеток.
    opacity_image = PILImage.fromarray(opacity_cells)
    opacity_image = opacity_image.resize((width, height), PILImage.Resampling.NEAREST)

    # Накладываем бирюзовый цвет там, где маска больше нуля.
    turquoise_image = PILImage.new("RGB", (width, height), (20, 185, 225))
    basemap.paste(turquoise_image, (0, 0), opacity_image)

    # Подготавливаем рисование меток поверх полупрозрачных дорожных клеток.
    image_draw = ImageDraw.Draw(basemap)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 12)

    # Переводим границы каждой клетки-метки в пиксели подложки и рисуем квадрат с подписью.
    for (row, column), color, label in map_markers:
        cell_left, cell_right = round(column/cells_per_side*width), round((column+1)/cells_per_side*width)-1
        cell_top, cell_bottom = round(row/cells_per_side*height), round((row+1)/cells_per_side*height)-1
        image_draw.rectangle((cell_left, cell_top, cell_right, cell_bottom), fill=tuple(color), outline="black", width=1)
        label_left = cell_right + 3
        # Старты подписываем слева; подпись у правого края тоже переносим влево.
        if (row, column) in start_cells or cell_left > width - 35:
            label_left = max(0, cell_left - 24)
        label_top = min(height-16, max(23, cell_top-6))
        image_draw.text((label_left, label_top), label, font=font, fill="black", stroke_width=2, stroke_fill="white")

    # Добавляем сверху объяснение цветов и сохраняем наложение.
    image_draw.rectangle((0, 0, width, 22), fill="white")
    image_draw.text((5, 4), "Бирюзовый: дорога | зелёный: старт | оранжевый: цель | синий: источники", font=font, fill="black")
    basemap.save(data_dir / "overlay_map.png")


# Создаём карту по настройкам в начале файла и сохраняем её в data.
def build_map():

    # Переводим расстояние от центра до края из метров в градусы широты и долготы.
    latitude_delta = MAP_HALF_SIDE_METERS / 111_320
    longitude_delta = MAP_HALF_SIDE_METERS / (111_320 * math.cos(math.radians(CENTER_LATITUDE)))

    # Границы участка: минимальная долгота, минимальная широта, максимальная долгота и широта.
    bounds_lon_lat = (
        CENTER_LONGITUDE - longitude_delta, CENTER_LATITUDE - latitude_delta,
        CENTER_LONGITUDE + longitude_delta, CENTER_LATITUDE + latitude_delta,
    )

    # Собираем подложку из сохранённых частей карты OSM.
    save_basemap(bounds_lon_lat, DATA_DIR)

    # Скачиваем линии OSM только при отсутствии сохранённого XML.
    osm_xml_path = DATA_DIR / "pavlodar.osm.xml"
    if not osm_xml_path.exists():
        response = requests.get(
            "https://api.openstreetmap.org/api/0.6/map",
            params={"bbox": ",".join(map(str, bounds_lon_lat))},
            headers={"User-Agent": "rl-uni-coursework-map/1.0 (student project)"}, timeout=90,
        )
        # При ошибке сервера останавливаем загрузку, чтобы не сохранить неверные данные.
        response.raise_for_status()
        osm_xml_path.write_bytes(response.content)

    # Строим клеточную сеть; старты и цель заданы в начале файла.
    road_grid, road_paths = create_road_grid(osm_xml_path, bounds_lon_lat, MAP_CELLS_PER_SIDE)
    start_cells = START_CELLS
    goal_cell = GOAL_CELL

    # Собираем клетки дорог, кроме стартов и цели.
    available_road_cells = []
    # Перебираем каждую клетку дороги.
    for row, column in np.argwhere(road_grid):
        cell = (int(row), int(column))
        if cell not in start_cells and cell != goal_cell:
            available_road_cells.append(cell)

    # Выбираем источники по seed и сортируем для одинаковой нумерации на картинках.
    random_generator = np.random.default_rng(TRAFFIC_SOURCE_SEED)
    chosen_indices = random_generator.choice(len(available_road_cells), TRAFFIC_SOURCE_COUNT, replace=False)
    traffic_sources = []
    # Переводим выбранные номера в координаты клеток.
    for index in chosen_indices:
        traffic_sources.append(available_road_cells[index])
    traffic_sources.sort()

    # Проверяем заданные вручную старты и цель; источники уже взяты с дороги.
    for cell in start_cells + [goal_cell]:
        if not road_grid[cell]:
            raise ValueError(f"Точка {cell} не попала на дорогу")

    # Кодируем клетки для файла задания: дорога 0, препятствие 1.
    encoded_map = np.where(road_grid, "0", "1")

    # Отмечаем все три старта кодом 2.
    for cell in start_cells:
        encoded_map[cell] = "2"

    # Отмечаем источники машин кодом 3.
    for cell in traffic_sources:
        encoded_map[cell] = "3"

    # Отмечаем цель кодом 4 и сохраняем строки сетки в текстовый файл.
    encoded_map[goal_cell] = "4"
    (DATA_DIR / "citymap.txt").write_text(
        "\n".join("".join(row) for row in encoded_map) + "\n", encoding="utf-8")

    # Создаём изображения той же карты и возвращаем сетку.
    save_map_images(road_grid, road_paths, bounds_lon_lat, start_cells, goal_cell,
                    traffic_sources, CENTER_LATITUDE, DATA_DIR)
    return road_grid


if __name__ == "__main__":
    # Создаём файлы карты, которые затем читает ноутбук.
    road_grid = build_map()
    print("Карта сохранена в:", DATA_DIR)
    print("Размер:", road_grid.shape, "дорожных клеток:", road_grid.sum())
