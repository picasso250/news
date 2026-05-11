import sys, os
from PIL import Image

def to_jpg(src: str, quality: int = 85):
    """Convert PNG to progressive JPEG q85"""
    img = Image.open(src)
    dst = src.rsplit('.', 1)[0] + '.jpg'
    if img.mode == 'RGBA':
        rgb = Image.new('RGB', img.size, (245, 240, 230))
        rgb.paste(img, mask=img.split()[3])
        img = rgb
    img.save(dst, 'JPEG', quality=quality, progressive=True, optimize=True)
    old = os.path.getsize(src)
    new = os.path.getsize(dst)
    print(f'{src} -> {dst}: {old/1024:.0f}KB -> {new/1024:.0f}KB ({new/old*100:.0f}%)')

if __name__ == '__main__':
    for f in sys.argv[1:]:
        to_jpg(f)
