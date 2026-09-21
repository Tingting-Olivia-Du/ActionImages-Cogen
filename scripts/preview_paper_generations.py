"""Build inspection previews from saved lossless predictions, without model inference."""
from pathlib import Path
import argparse
import json
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path)
    args=p.parse_args()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for metadata in sorted(args.source.glob('figure*/generation.json')):
        directory=metadata.parent
        info=json.loads(metadata.read_text())
        shown=info['shown_indices']
        preview=directory/'generated_two_views.mp4'
        if not preview.exists():
            with np.load(directory/'frames.npz') as data:
                perception,action=data['perception'],data['action']
                frames=[np.concatenate([np.concatenate([perception[v,i],action[v,i]],axis=1)
                        for v in range(2)],axis=0) for i in range(41)]
                imageio.mimwrite(preview,frames,fps=20/3,quality=8,macro_block_size=16)
        sheet=Image.new('RGB',(5*256,2*286),'white')
        draw=ImageDraw.Draw(sheet)
        cells=[('gt_perception',0),('perception',shown[0]),('perception',shown[1]),
               ('action',shown[0]),('action',shown[1])]
        for view in [1,2]:
            for col,(kind,i) in enumerate(cells):
                title='Observed' if kind.startswith('gt') else f'{kind} +{i*3/20:.2f}s'
                draw.text((col*256+4,(view-1)*286+5),f'v{view}: {title}',fill='black',font=font)
                im=Image.open(directory/f'view{view}/{kind}/{i:03d}.png')
                sheet.paste(im.resize((256,256),Image.Resampling.LANCZOS),(col*256,(view-1)*286+30))
        sheet.save(directory/'quicklook.png')
        print('PREVIEW',directory.name,flush=True)


if __name__=='__main__':
    main()
