from app.pipeline import padded_box


def test_padded_box_expands_within_bounds():
    box = padded_box((100.0, 100.0, 200.0, 300.0), img_w=1000, img_h=1000, pad_frac=0.1)
    assert box == (90.0, 80.0, 210.0, 320.0)


def test_padded_box_clips_at_image_edges():
    box = padded_box((0.0, 0.0, 50.0, 50.0), img_w=1000, img_h=1000, pad_frac=0.5)
    assert box[0] == 0.0
    assert box[1] == 0.0

    box2 = padded_box((950.0, 950.0, 1000.0, 1000.0), img_w=1000, img_h=1000, pad_frac=0.5)
    assert box2[2] == 1000.0
    assert box2[3] == 1000.0


def test_padded_box_zero_size_box_does_not_blow_up():
    box = padded_box((100.0, 100.0, 100.0, 100.0), img_w=1000, img_h=1000, pad_frac=0.15)
    assert box == (100.0, 100.0, 100.0, 100.0)
