from pathlib import Path
import re,os

filelist = list(Path('.').rglob('*.png'))


def patient_info(f):
    filename: str = Path(f).stem if isinstance(f, Path) else f.stem
    pattern = re.compile('\d+')
    info = pattern.findall(filename)
    return info


def new_filename(patient_num, trail_num, slice_num):
    return f'patient{int(patient_num):03d}_{int(trail_num):02d}_{int(slice_num):02d}.png'

def parent_and_stem(path_complete, new_stem):
    return Path(str(path_complete)).parent / new_stem

for f in filelist:
    new_stem = new_filename(*patient_info(f))
    new_path = parent_and_stem(f, new_stem)
    print(f'{f}->{new_path}')
    os.rename(f,new_path)




