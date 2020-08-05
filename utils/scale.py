from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

WIDTH, HEIGHT = 224, 224


def main(args):
    out_path = Path(args.out_path)
    for idx, name in enumerate(args.files):
        filepath = Path(name)
        img = Image.open(name)
        img = img.resize((WIDTH, HEIGHT))
        # img.show()
        val, _, __ = str(filepath.stem).split("_")
        copy_name = "%d_%s.jpg" % (idx, val)
        # print(val, out_path / copy_name)
        img.save(out_path / copy_name)
    print("Done!")


if __name__ == "__main__":
    out_path = Path(__file__).absolute().parent.parent / 'static/files'
    parser = ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--out_path', default=str(out_path))
    args = parser.parse_args()
    main(args)
