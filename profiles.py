from collections import namedtuple

Profile = namedtuple('Profile', ['imgsize', 'max_scale', 'maps'])
MapParams = namedtuple('MapParams', ['size', 'scale', 'n_bboxes', 'ratios'])

voc_ssd_300 = Profile(imgsize=(300, 300), max_scale=1.0,
                      maps=[MapParams((38, 38), 0.2, 4, [1.0, 2.0, 1.0 / 2.0]),
                            MapParams((19, 19), 0.34, 6, [1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0]),
                            MapParams((10, 10), 0.48, 6, [1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0]),
                            MapParams((5, 5), 0.62, 6, [1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0]),
                            MapParams((3, 3), 0.76, 6, [1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0]),
                            MapParams((1, 1), 0.9, 6, [1.0, 2.0, 1.0 / 2.0, 3.0, 1.0 / 3.0])])
