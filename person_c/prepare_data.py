"""Download CUB images and Reed captions; never executes third-party dataset code."""
import hashlib
import tarfile
from pathlib import Path

import gdown
import requests

ROOT = Path('/content/person_c_data')
CUB_MD5 = '97eceeb196236b17998738112f37df78'
CAPTION_ID = '0B0ywwgffWnLLZW9uVHNjb2JmNlE'


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'CUB_200_2011.tgz'
    if not (root / 'CUB_200_2011/images.txt').exists():
        for url in [
            'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1',
            'https://data.caltech.edu/api/records/65de6-vp158/files/CUB_200_2011.tgz/content',
        ]:
            with requests.get(url, stream=True, timeout=60) as response:
                print(response.status_code, url, flush=True)
                if not response.ok: continue
                with archive.open('wb') as output:
                    for chunk in response.iter_content(1024 * 1024): output.write(chunk)
                break
        if not archive.exists(): raise RuntimeError('CUB download unavailable; check source access.')
        with archive.open('rb') as source:
            digest = hashlib.file_digest(source, 'md5').hexdigest()
        if digest != CUB_MD5: raise ValueError('CUB checksum mismatch')
        with tarfile.open(archive) as source: source.extractall(root, filter='data')
    if len(list((root / 'birds/text').rglob('*.txt'))) != 11788:
        archive = root / 'reed_cub.tar.gz'
        if not archive.exists():
            gdown.download(id=CAPTION_ID, output=str(archive), quiet=False)
        with tarfile.open(archive) as source:
            for member in source:
                parts = Path(member.name).parts
                if member.isfile() and 'text_c10' in parts and member.name.endswith('.txt'):
                    suffix = Path(*parts[parts.index('text_c10') + 1:])
                    if '..' in suffix.parts: raise ValueError('Invalid archive path')
                    target = root / 'birds/text' / suffix
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.extractfile(member).read())
    images = list((root / 'CUB_200_2011/images').rglob('*.jpg'))
    captions = list((root / 'birds/text').rglob('*.txt'))
    print('Images:', len(images), 'caption files:', len(captions), flush=True)
    if len(images) != 11788 or len(captions) != 11788: raise ValueError('Incomplete dataset')


if __name__ == '__main__':
    prepare()
