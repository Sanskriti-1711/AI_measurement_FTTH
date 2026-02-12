import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def autodetect_scaniverse(root: str):
    for name in os.listdir(root):
        if name.startswith('Scaniverse'):
            inner = os.path.join(root, name, name)
            if os.path.isdir(inner):
                return inner
            return os.path.join(root, name)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default=None, help='Path to Scaniverse folder or archive (overrides auto-detect)')
    p.add_argument('--out-dir', '-o', default=os.path.join(ROOT, 'out'), help='Output directory')
    args = p.parse_args()

    target = args.input
    if target is None:
        target = autodetect_scaniverse(ROOT)
        if target is None:
            print('Scaniverse folder not found under', ROOT)
            sys.exit(2)

    if not os.path.exists(target):
        # try to match by substring under ROOT (helpful for paths with special spaces)
        basename = os.path.basename(target)
        candidate = None
        for name in os.listdir(ROOT):
            if basename in name or ('Scaniverse' in (basename or '') and name.startswith('Scaniverse')):
                candidate = os.path.join(ROOT, name)
                break
        if candidate and os.path.exists(candidate):
            print('Input path not found; using matched folder:', candidate)
            target = candidate
        else:
            print('Provided input path does not exist:', target)
            sys.exit(2)

    print('Using model input folder:', target)

    out_json = os.path.join(args.out_dir, 'scaniverse_dims.json')
    out_csv = os.path.join(args.out_dir, 'scaniverse_dims.csv')
    sys.argv = ['model_dims.py', '--input', target, '--out-json', out_json, '--out-csv', out_csv]

    import model_dims
    model_dims.main()

    print('Wrote:', out_json, out_csv)


if __name__ == '__main__':
    main()
