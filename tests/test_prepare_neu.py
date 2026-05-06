import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_neu.py"
SPEC = importlib.util.spec_from_file_location("prepare_neu", SCRIPT_PATH)
prepare_neu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare_neu)


def test_voc_box_to_yolo_conversion():
    x_center, y_center, width, height = prepare_neu.voc_box_to_yolo(50, 25, 150, 125, 200, 200)

    assert x_center == 0.5
    assert y_center == 0.375
    assert width == 0.5
    assert height == 0.5


def test_xml_conversion_uses_annotation_image_size():
    xml = """\
<annotation>
  <size><width>400</width><height>200</height></size>
  <object>
    <name>scratches</name>
    <bndbox>
      <xmin>100</xmin><ymin>50</ymin><xmax>300</xmax><ymax>150</ymax>
    </bndbox>
  </object>
</annotation>
"""
    with TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "sample.xml"
        label_path = Path(tmp) / "sample.txt"
        xml_path.write_text(xml, encoding="utf-8")

        prepare_neu.convert_xml_annotation(xml_path, label_path)

        assert label_path.read_text(encoding="utf-8").strip() == "5 0.500000 0.500000 0.500000 0.500000"


def test_stratified_fold_creation_keeps_class_balance_shape():
    image_names = [
        "crazing_1.jpg",
        "crazing_2.jpg",
        "crazing_3.jpg",
        "crazing_4.jpg",
        "scratches_1.jpg",
        "scratches_2.jpg",
        "scratches_3.jpg",
        "scratches_4.jpg",
    ]
    folds = prepare_neu.create_stratified_folds(image_names, k=4, seed=244)

    assert len(folds) == 4
    assert all(len(fold) == 2 for fold in folds)
    assert sorted(name for fold in folds for name in fold) == sorted(image_names)
