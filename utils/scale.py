from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

WIDTH, HEIGHT = 224, 224


def main(args):
    out_path = Path(args.out_path)
    for name in args.files:
        filepath = Path(name)
        img = Image.open(name)
        img = img.resize((WIDTH, HEIGHT))
        img.show()
        img.save(out_path / filepath.name)
    print("Done!")


if __name__ == "__main__":
    out_path = Path(__file__).absolute().parent.parent / 'static/files'
    parser = ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--out_path', default=str(out_path))
    args = parser.parse_args()
    main(args)