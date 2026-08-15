from app.shelf_clustering import BoxLike, assign_shelves


def test_empty_detections():
    assert assign_shelves([]) == ([], 0)


def test_single_detection():
    assert assign_shelves([BoxLike((0, 0, 10, 10))]) == ([1], 1)


def test_three_well_separated_rows():
    dets = [
        BoxLike((0, 10, 50, 60)),
        BoxLike((100, 15, 150, 65)),
        BoxLike((0, 300, 50, 350)),
        BoxLike((100, 305, 150, 355)),
        BoxLike((0, 600, 50, 650)),
    ]
    shelf_ids, count = assign_shelves(dets)
    assert shelf_ids == [1, 1, 2, 2, 3]
    assert count == 3


def test_same_shelf_different_heights_aligned_bottoms():
    dets = [
        BoxLike((0, 10, 50, 400)),   # tall bottle
        BoxLike((100, 380, 150, 400)),  # short item, same bottom edge
    ]
    shelf_ids, count = assign_shelves(dets)
    assert shelf_ids == [1, 1]
    assert count == 1
