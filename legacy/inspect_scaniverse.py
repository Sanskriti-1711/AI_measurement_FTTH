import os
p = r'D:\Downloads_D\Q-GIS\Aruco_marker'
for name in os.listdir(p):
    if name.startswith('Scaniverse'):
        print('NAME_REPR:', repr(name))
        print('CODEPOINTS:', [hex(ord(c)) for c in name])
        print('PATH:', os.path.join(p, name))
